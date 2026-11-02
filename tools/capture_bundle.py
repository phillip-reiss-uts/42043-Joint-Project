#!/usr/bin/env python3
"""Capture one hardware dataset for offline identification of the state-space
observer. Nothing is fitted or tuned here.

Phases, in one connection:

    still    hold the start pose. Gives encoder noise (R), qd noise, and the
             real cycle time.
    step     each joint, +STEP then back to the start pose. Both directions of
             every joint, so friction hysteresis is visible.
    chirp    a linear frequency sweep on two joints. Steps give one time
             constant; the sweep gives a frequency response, which is what
             distinguishes a delay from a lag.

Writes <out>.csv (one row per control cycle: time, phase, q, qd, command) and
<out>.json (start pose, per-phase plan, config snapshot, achieved rate).

At home, replay it with tools/validate_state_space.py fit, or fit the delay and
noise in your own script -- the CSV has the raw command and real timestamps.

    python tools/capture_bundle.py --out ../notes/hw_bundle
    python tools/capture_bundle.py --out ../notes/hw_bundle --no-chirp

Safety: 2 deg steps, 0.30 rad/s joint-speed limit, per-phase timeout, one
Enter to start, nothing retried. Pendant at 10%, cell clear, hand on the stop.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg
from src.run_log import start_run_log
from src.robot_backend_ur import connect

STILL_S = 20.0
STEP_DEG = 2.0
STEP_S = 3.0
CHIRP_S = 12.0
CHIRP_F0 = 0.2
CHIRP_F1 = 3.0
CHIRP_DEG = 2.0
CHIRP_JOINTS = (1, 3)
QD_MAX = 0.30
PER_CYCLE_BUDGET_S = 0.040


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(path, rows):
    _ensure_parent(path)
    n = cfg.ARM_DOF
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t', 'phase', 't_cmd']
                   + ['q%d' % i for i in range(n)]
                   + ['qd%d' % i for i in range(n)]
                   + ['u%d' % i for i in range(n)])
        for r in rows:
            w.writerow([r['t'], r['phase'], r['t_cmd']]
                       + list(r['q']) + list(r['qd']) + list(r['u']))


class Capture:
    def __init__(self, backend, q0):
        self.b = backend
        self.q0 = np.asarray(q0, float).copy()
        self.rows = []
        self.t0 = time.monotonic()
        self.abort = None
        self.phase = ''
        self.phase_t0 = self.t0
        self.phase_budget = 0.0

    def set_phase(self, name, budget_s):
        self.phase = name
        self.phase_t0 = time.monotonic()
        self.phase_budget = float(budget_s)

    def elapsed(self):
        return time.monotonic() - self.t0

    def cycle(self, u):
        # One control cycle: stream, block for the controller, read back.
        # t_cmd and t are both logged because they differ by the servoJ call
        # and the readback -- that difference is the loop's real latency.
        tc = time.monotonic()
        self.b.command(np.asarray(u, float), cfg.GRIP_OPEN)
        self.b.step()
        q = np.asarray(self.b.joint_positions(), float)
        qd = np.asarray(self.b.joint_velocities(), float)
        tr = time.monotonic()
        self.rows.append(dict(t=tr - self.t0, phase=self.phase,
                              t_cmd=tc - self.t0, q=q.copy(), qd=qd.copy(),
                              u=np.asarray(u, float).copy()))
        if np.max(np.abs(qd)) > QD_MAX:
            self.abort = 'joint speed %.2f rad/s in %s' % (
                float(np.max(np.abs(qd))), self.phase)
        elif tr - self.phase_t0 > self.phase_budget:
            self.abort = 'phase %s exceeded %.1f s' % (self.phase,
                                                       self.phase_budget)
        return q, qd


def _budget(nominal_s, use_rate):
    # servoJ blocks past dt on this build (~22 ms measured), so a timeout sized
    # from the nominal rate aborts a good capture.
    return nominal_s * max(1.0, use_rate / cfg.UR_CYCLE_DT) + 2.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default=cfg.UR_IP)
    ap.add_argument('--out', default=None)
    ap.add_argument('--joints', default='0,1,2,3,4,5',
                    help='comma-separated joint indices for the step phase')
    ap.add_argument('--no-chirp', action='store_true')
    ap.add_argument('--gain', type=float, default=cfg.UR_SERVO_GAIN)
    ap.add_argument('--lookahead', type=float, default=cfg.UR_LOOKAHEAD_TIME)
    args = ap.parse_args()
    start_run_log(sys.argv[1:], ROOT, tag='capture_bundle')

    joints = [int(v) for v in args.joints.split(',') if v.strip()]
    for j in joints:
        if not 0 <= j < cfg.ARM_DOF:
            raise SystemExit('joint %d out of range' % j)
    if abs(STEP_DEG) > 3.0 or abs(CHIRP_DEG) > 3.0:
        raise SystemExit('steps above 3 deg are refused')

    print('CAPTURE BUNDLE: still, per-joint steps, chirp. No tuning.')
    print('Pendant 10%, cell clear, no cube, hand on the stop.')
    input('Press Enter to confirm > ')

    backend = connect(args.ip, activate_gripper=False, dt=cfg.UR_CYCLE_DT,
                      lookahead_time=args.lookahead,
                      gain=args.gain)
    cap = None
    try:
        q0 = np.asarray(backend.joint_positions(), float).copy()
        cap = Capture(backend, q0)

        # ---- still: noise, and the real cycle time ---------------------
        print('  still %.0f s' % STILL_S)
        cap.set_phase('still', _budget(STILL_S, 0.0))
        end = time.monotonic() + STILL_S
        while time.monotonic() < end and cap.abort is None:
            cap.cycle(q0)

        # ---- steps: every joint, up then back (the reverse step) --------
        for j in joints:
            up = q0.copy()
            up[j] += np.radians(STEP_DEG)
            for name, u in (('step_j%d_up' % j, up),
                            ('step_j%d_dn' % j, q0)):
                if cap.abort is not None:
                    break
                print('  %s %.0f s' % (name, STEP_S))
                cap.set_phase(name, _budget(STEP_S, 0.0))
                end = time.monotonic() + STEP_S
                while time.monotonic() < end and cap.abort is None:
                    cap.cycle(u)

        # ---- chirp: frequency content, which separates delay from lag ---
        if not args.no_chirp and cap.abort is None:
            for j in CHIRP_JOINTS:
                if j >= cfg.ARM_DOF:
                    continue
                print('  chirp j%d %.1f-%.1f Hz %.0f s'
                      % (j, CHIRP_F0, CHIRP_F1, CHIRP_S))
                cap.set_phase('chirp_j%d' % j, _budget(CHIRP_S, 0.0))
                t_start = time.monotonic()
                while cap.abort is None:
                    el = time.monotonic() - t_start
                    if el >= CHIRP_S:
                        break
                    # linear sweep; phase = 2*pi*(f0 t + (f1-f0) t^2 / 2T)
                    ph = 2.0 * np.pi * (CHIRP_F0 * el
                                        + (CHIRP_F1 - CHIRP_F0)
                                        * el * el / (2.0 * CHIRP_S))
                    u = q0.copy()
                    u[j] += np.radians(CHIRP_DEG) * np.sin(ph)
                    cap.cycle(u)
    finally:
        try:
            if cap is not None:
                backend.command(cap.q0, cfg.GRIP_OPEN)
                for _ in range(int(0.5 / cfg.UR_CYCLE_DT)):
                    backend.step()
        except Exception:
            pass
        backend.close()

    base = args.out or os.path.join(
        ROOT, 'hw_bundle_%s' % time.strftime('%Y-%m-%d_%H%M'))
    base = os.path.splitext(base)[0]

    if not cap.rows:
        print('no rows captured')
        return 2
    t = np.array([r['t'] for r in cap.rows])
    phases = sorted(set(r['phase'] for r in cap.rows))
    dt_mean = float(np.mean(np.diff(t))) if len(t) > 1 else float('nan')
    qd_noise = {}
    still = [r for r in cap.rows if r['phase'] == 'still']
    if len(still) > 10:
        qd = np.array([r['qd'] for r in still])
        qd_noise = dict(zip(('qd%d' % i for i in range(cfg.ARM_DOF)),
                            map(float, np.std(qd, axis=0))))
    _write_csv(base + '.csv', cap.rows)
    meta = dict(q_start=list(map(float, cap.q0)),
                target_dt=float(cfg.UR_CYCLE_DT),
                dt_measured=dt_mean,
                rate_hz=float(1.0 / dt_mean) if dt_mean else None,
                phases=phases,
                rows=len(cap.rows),
                aborted=cap.abort,
                still_qd_std=qd_noise,
                config=dict(UR_SERVO_KP=cfg.UR_SERVO_KP,
                            UR_SERVO_KV=cfg.UR_SERVO_KV,
                            UR_LOOKAHEAD_TIME=cfg.UR_LOOKAHEAD_TIME,
                            UR_SERVO_GAIN=cfg.UR_SERVO_GAIN,
                            UR_CYCLE_DT=cfg.UR_CYCLE_DT),
                created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    _ensure_parent(base + '.json')
    with open(base + '.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    print()
    print('wrote %s.csv / .json  (%d rows, %d phases)'
          % (base, len(cap.rows), len(phases)))
    print('achieved %.1f Hz (dt %.1f ms)' % (1.0 / dt_mean, dt_mean * 1e3))
    if qd_noise:
        print('still qd std (rad/s): %s'
              % ' '.join('%.4f' % v for v in qd_noise.values()))
    if cap.abort:
        print('ABORTED: %s -- the record is still written, but a phase is '
              'truncated' % cap.abort)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
