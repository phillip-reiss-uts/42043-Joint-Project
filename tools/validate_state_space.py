#!/usr/bin/env python3
"""Validate and fit the hardware state-space observer/controller.

Three modes:

  record   One small joint step on the real arm, logging q, qdot, the raw
           command u and the observer estimate. No LQR correction is applied.
  fit      Offline replay of that log for candidate UR_SERVO_KP/KV values.
           Writes a fit JSON; --write updates config.
  ab       On the real arm, runs the same joint step raw and state-space and
           compares them. Refuses a fit whose qdot residual is too high.

The A/B is required: without it the state-space path is only an assertion.

Safety limits here are independent of the gains being tested:
    max step      3 deg
    max joint qd  0.30 rad/s
    timeout       step + hold + 2 s
Nothing is retried after an abort.

    python tools/validate_state_space.py record --joint 1 --step 1.5
    python tools/validate_state_space.py fit --record <file>.csv --write
    python tools/validate_state_space.py ab --fit <fit>.json --joint 1 --step 1.5
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg
from src.run_log import start_run_log
from src.arm import Arm
from src.arm_dynamics import ArmDynamics
from src.controller import Encoder, LQRController, lqr_weights
from src.kinematics_ur import URKinematics
from src.observer import ArmObserver
from src.robot_backend_ur import connect

CONFIG_PATH = os.path.join(ROOT, 'src', 'config.py')
CAL_STEP_MAX_DEG = 3.0
CAL_QD_MAX = 0.30
CAL_HOLD_S = 2.0
FIT_MAX_RMS = 0.05          # rad/s, qdot residual
AB_FINAL_DEG = 0.30
AB_OVER_DEG = 0.50
AB_SETTLE_S = 1.20


def _kin(ip=None, kp=None, kv=None, dyn=None):
    dyn = dyn or ArmDynamics.from_file(
        os.path.join(ROOT, 'models', 'ur3e', 'arm_dynamics.json'))
    kp = cfg.UR_SERVO_KP if kp is None else float(kp)
    kv = cfg.UR_SERVO_KV if kv is None else float(kv)
    return URKinematics(dyn, (np.full(cfg.ARM_DOF, kp),
                             np.full(cfg.ARM_DOF, kv)),
                        tool_pos=(0.0, 0.0, 0.0),
                        tool_quat=(1.0, 0.0, 0.0, 0.0),
                        root_yaw=cfg.UR_ROOT_YAW)


def _target(q0, joint, step_deg):
    if not 0 <= joint < cfg.ARM_DOF:
        raise ValueError('joint must be 0..5')
    step = float(step_deg)
    if abs(step) > CAL_STEP_MAX_DEG:
        raise ValueError('step above %.1f deg' % CAL_STEP_MAX_DEG)
    q = np.asarray(q0, float).copy()
    q[joint] += np.radians(step)
    return q


def _columns(prefix, n=cfg.ARM_DOF):
    return ['%s%d' % (prefix, i) for i in range(n)]


def _ensure_parent(path):
    # --out notes/x is relative to the CWD, and the notes repo lives outside
    # this tree; creating the parent turns a half-finished 2.5 s hardware run
    # into a written record instead of a FileNotFoundError at the end.
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _csv_write(path, rows):
    qc = _columns('q')
    vc = _columns('qd')
    uc = _columns('u')
    hc = _columns('qhat')
    dvc = _columns('qdhat')
    _ensure_parent(path)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t'] + qc + vc + uc + hc + dvc)
        for r in rows:
            w.writerow([r['t']]
                       + list(r['q']) + list(r['qd']) + list(r['u'])
                       + list(r['qhat']) + list(r['qdhat']))


def _csv_read(path):
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            rows.append({k: float(v) if k != 't' else float(v)
                         for k, v in r.items()})
    return rows


def _load_record(path):
    meta_path = os.path.splitext(path)[0] + '.json'
    with open(meta_path) as fh:
        meta = json.load(fh)
    return meta, _csv_read(path)


def _write_config(kp, kv):
    src = open(CONFIG_PATH).read()
    new, n1 = re.subn(r'^UR_SERVO_KP = .*$', 'UR_SERVO_KP = %.3f' % kp,
                      src, count=1, flags=re.MULTILINE)
    new, n2 = re.subn(r'^UR_SERVO_KV = .*$', 'UR_SERVO_KV = %.3f' % kv,
                      new, count=1, flags=re.MULTILINE)
    if n1 != 1 or n2 != 1:
        print('  could not find UR_SERVO_KP/KV in config -- NOT written')
        return False
    open(CONFIG_PATH, 'w').write(new)
    print('  wrote %s' % CONFIG_PATH)
    return True


def _time_limit(n):
    # servoJ blocks past dt on this build (~22 ms/cycle measured), so a timeout
    # sized from the NOMINAL rate aborts a perfectly good step. Allow 40 ms per
    # cycle, which is still a watchdog for a stuck loop.
    return max(CAL_HOLD_S + 2.0, 0.040 * int(n) + 1.0)


def _safety(qd, t0, limit_s):
    if float(np.max(np.abs(qd))) > CAL_QD_MAX:
        return 'joint speed %.2f rad/s' % float(np.max(np.abs(qd)))
    if time.monotonic() - t0 > limit_s:
        return 'timeout'
    return None


def cmd_record(args):
    print('RECORD: raw joint step, observer prediction logged only.')
    print('Pendant 10%, cell clear, no cube, hand on the stop.')
    input('Press Enter to confirm > ')
    backend = connect(args.ip, activate_gripper=False, dt=cfg.UR_CYCLE_DT,
                      lookahead_time=cfg.UR_LOOKAHEAD_TIME,
                      gain=cfg.UR_SERVO_GAIN)
    kin = _kin()
    arm = Arm(kinematics=kin, backend=backend)
    rows = []
    aborted = None
    q0 = np.asarray(backend.joint_positions(), float).copy()
    u = _target(q0, args.joint, args.step)
    obs = ArmObserver(None, None, q0, dt=cfg.UR_CYCLE_DT, kin=kin)
    t0 = time.monotonic()
    n = int((CAL_HOLD_S + 0.5) / cfg.UR_CYCLE_DT)
    try:
        for i in range(n):
            obs.predict(u)
            backend.command(u, cfg.GRIP_OPEN)
            backend.step()
            q = np.asarray(backend.joint_positions(), float)
            qd = np.asarray(backend.joint_velocities(), float)
            obs.update(q)
            rows.append(dict(t=time.monotonic() - t0, q=q.copy(), qd=qd.copy(),
                             u=u.copy(), qhat=obs.xhat[:cfg.ARM_DOF].copy(),
                             qdhat=obs.xhat[cfg.ARM_DOF:].copy()))
            aborted = _safety(qd, t0, _time_limit(n))
            if aborted:
                break
    finally:
        backend.close()

    dt_meas = float(rows[-1]['t'] - rows[0]['t']) / max(1, len(rows) - 1) \
        if len(rows) > 1 else float(cfg.UR_CYCLE_DT)
    print('achieved %.1f Hz (dt %.1f ms) over %d rows'
          % (1.0 / dt_meas, dt_meas * 1e3, len(rows)))
    base = args.out or os.path.join(
        ROOT, 'state_space_record_%s' % time.strftime('%Y-%m-%d_%H%M'))
    base = os.path.splitext(base)[0]
    _csv_write(base + '.csv', rows)
    meta = dict(q0=list(map(float, q0)), dt=dt_meas,
                target_dt=float(cfg.UR_CYCLE_DT),
                joint=int(args.joint), step_deg=float(args.step),
                kp=float(cfg.UR_SERVO_KP), kv=float(cfg.UR_SERVO_KV),
                aborted=aborted, n=len(rows),
                created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    _ensure_parent(base + '.json')
    with open(base + '.json', 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('wrote %s.csv / .json  (%d rows)' % (base, len(rows)))
    if aborted:
        print('ABORTED: %s -- do not use this record' % aborted)
        return 2
    return 0


def _fit_score(meta, rows, kp, kv):
    # A gain pair can be UNSTABLE at the achieved timestep: at 45 Hz a high
    # Kv relative to Kp makes the discretised closed loop diverge, the state
    # goes non-finite, and the DARE inside the observer raises. Such a pair is
    # simply not a candidate -- score it as unusable rather than crashing the
    # whole fit on the first bad corner of the grid.
    try:
        kin = _kin(kp=kp, kv=kv)
        obs = ArmObserver(None, None, np.asarray(meta['q0'], float),
                          dt=float(meta['dt']), kin=kin)
        errs = []
        for r in rows:
            q = np.array([r[c] for c in _columns('q')])
            qd = np.array([r[c] for c in _columns('qd')])
            u = np.array([r[c] for c in _columns('u')])
            obs.predict(u)
            obs.update(q)
            errs.append(obs.xhat[cfg.ARM_DOF:] - qd)
        e = np.asarray(errs)
        if not np.all(np.isfinite(e)):
            return float('inf')
        return float(np.sqrt(np.mean(e * e)))
    except Exception:
        return float('inf')


def cmd_fit(args):
    meta, rows = _load_record(args.record)
    if meta.get('aborted'):
        raise SystemExit('record was aborted; not fitting it')
    if not rows:
        raise SystemExit('record is empty')
    kp0 = float(meta.get('kp', cfg.UR_SERVO_KP))
    kv0 = float(meta.get('kv', cfg.UR_SERVO_KV))
    kps = np.geomspace(kp0 / 8.0, kp0 * 8.0, 7)
    kvs = np.geomspace(kv0 / 8.0, kv0 * 8.0, 7)
    best = None
    print('replaying %d rows over %dx%d candidates' % (len(rows), len(kps), len(kvs)))
    for kp in kps:
        for kv in kvs:
            rms = _fit_score(meta, rows, kp, kv)
            if best is None or rms < best[0]:
                best = (rms, float(kp), float(kv))
    print('best RMS %.4f rad/s at Kp %.1f Kv %.1f' % best)
    passed = best[0] <= FIT_MAX_RMS
    out = dict(kp=best[1], kv=best[2], rms=best[0],
               record=os.path.abspath(args.record),
               passed=passed, fit_max_rms=FIT_MAX_RMS,
               joint=meta.get('joint'), step_deg=meta.get('step_deg'))
    path = args.out or os.path.splitext(args.record)[0] + '_fit.json'
    _ensure_parent(path)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('wrote %s' % path)
    if not passed:
        print('FAIL: qdot residual above %.3f rad/s; do not enable state-space'
              % FIT_MAX_RMS)
        return 2
    if args.write:
        _write_config(best[1], best[2])
    return 0


def _run_step(backend, arm, q0, u, state_space, kin):
    if state_space:
        obs = ArmObserver(None, None, q0, dt=cfg.UR_CYCLE_DT, kin=kin,
                          lqr_weights=lqr_weights())
        arm.attach_observer(obs, LQRController(obs), Encoder(noisy=False))
    else:
        arm.detach_observer()

    rows = []
    aborted = None
    t0 = time.monotonic()
    n = int((CAL_HOLD_S + 0.5) / cfg.UR_CYCLE_DT)
    try:
        for i in range(n):
            if state_space:
                arm.command_step(u, cfg.GRIP_OPEN)
            else:
                arm.command_step(u, cfg.GRIP_OPEN)
            q = np.asarray(backend.joint_positions(), float)
            qd = np.asarray(backend.joint_velocities(), float)
            rows.append((i * cfg.UR_CYCLE_DT, q.copy(), qd.copy()))
            aborted = _safety(qd, t0, _time_limit(n))
            if aborted:
                break
    finally:
        arm.detach_observer()
    return rows, aborted


def _metrics(rows, q0, u, joint):
    if not rows:
        return None
    q = np.asarray([r[1] for r in rows])
    qd = np.asarray([r[2] for r in rows])
    err = np.abs(q[:, joint] - u[joint])
    final = float(err[-1])
    step = u[joint] - q0[joint]
    direction = 1.0 if step >= 0 else -1.0
    over = float(np.max(np.maximum(0.0, (q[:, joint] - u[joint]) * direction)))
    tol = np.radians(0.10)
    bad = np.nonzero(err > tol)[0]
    settle = float(bad[-1] * cfg.UR_CYCLE_DT) if len(bad) else 0.0
    return dict(final_deg=float(np.degrees(final)),
                overshoot_deg=float(np.degrees(over)),
                settle_s=settle,
                peak_qd=float(np.max(np.abs(qd))))


def cmd_ab(args):
    with open(args.fit) as fh:
        fit = json.load(fh)
    if not fit.get('passed') or fit.get('rms', 1e9) > FIT_MAX_RMS:
        raise SystemExit('fit is not acceptable (RMS %.4f); refusing A/B'
                         % fit.get('rms', 1e9))
    print('A/B: raw vs state-space, same %.1f deg step on joint %d'
          % (args.step, args.joint))
    print('Pendant 10% first; clear cell; hand on the stop.')
    input('Press Enter to confirm > ')
    kp, kv = float(fit['kp']), float(fit['kv'])
    kin = _kin(kp=kp, kv=kv)
    backend = connect(args.ip, activate_gripper=False, dt=cfg.UR_CYCLE_DT,
                      lookahead_time=cfg.UR_LOOKAHEAD_TIME,
                      gain=cfg.UR_SERVO_GAIN)
    arm = Arm(kinematics=kin, backend=backend)
    try:
        q0 = np.asarray(backend.joint_positions(), float).copy()
        u = _target(q0, args.joint, args.step)
        raw_rows, raw_abort = _run_step(backend, arm, q0, u, False, kin)
        arm.move(q0, cfg.GRIP_OPEN, rate=0.03)
        state_rows, state_abort = _run_step(backend, arm, q0, u, True, kin)
    finally:
        backend.close()
    raw = _metrics(raw_rows, q0, u, args.joint)
    state = _metrics(state_rows, q0, u, args.joint)
    print('\n%-12s %10s %10s' % ('metric', 'raw', 'state-space'))
    for k in ('final_deg', 'overshoot_deg', 'settle_s', 'peak_qd'):
        print('%-12s %10.3f %10.3f' % (k, raw[k], state[k]))
    if raw_abort or state_abort:
        print('ABORTED: raw=%s state=%s' % (raw_abort, state_abort))
        return 2
    ok = (state['final_deg'] <= max(AB_FINAL_DEG,
                                   raw['final_deg'] + 0.10)
          and state['overshoot_deg'] <= AB_OVER_DEG
          and state['settle_s'] <= AB_SETTLE_S
          and state['peak_qd'] <= CAL_QD_MAX)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    rec = sub.add_parser('record')
    rec.add_argument('--ip', default=cfg.UR_IP)
    rec.add_argument('--joint', type=int, default=1)
    rec.add_argument('--step', type=float, default=1.5)
    rec.add_argument('--out', default=None)
    rec.set_defaults(fn=cmd_record)

    fit = sub.add_parser('fit')
    fit.add_argument('--record', required=True)
    fit.add_argument('--out', default=None)
    fit.add_argument('--write', action='store_true')
    fit.set_defaults(fn=cmd_fit)

    ab = sub.add_parser('ab')
    ab.add_argument('--fit', required=True)
    ab.add_argument('--ip', default=cfg.UR_IP)
    ab.add_argument('--joint', type=int, default=1)
    ab.add_argument('--step', type=float, default=1.5)
    ab.set_defaults(fn=cmd_ab)

    args = ap.parse_args()
    start_run_log(sys.argv[1:], ROOT, tag='validate_state_space')
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
