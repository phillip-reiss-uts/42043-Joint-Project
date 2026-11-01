#!/usr/bin/env python3
"""Pre-lab robustness bench for the state-space joint loop.

This is a controller-in-the-loop test, not a pick test. It wraps the MuJoCo
plant in a command delay + first-order lag that approximates the UR servoJ
lookahead, then runs the same small joint step with:

    observer Kp/Kv = plant Kp/Kv * 0.5, 1.0, 2.0
    command delay  = 0, 40, 80, 120 ms
    LQR gain scale = 0.25, 0.50, 1.00

It reports final error, overshoot, settling, peak joint speed and the observer's
qdot error. A case is marked stable only if all of those are bounded. The tool
passes only if every case at the configured LQR_GAIN_SCALE is stable.

The real controller still has to be fitted and A/B-tested. This bench answers:
is the LQR gain too aggressive for the delay and model mismatch we know about?

    python tools/bench_state_space.py
    python tools/bench_state_space.py --out notes/state_space_bench.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mujoco

from src import config as cfg
from src.arm import Arm
from src.arm_dynamics import ArmDynamics
from src.controller import Encoder, LQRController, lqr_weights
from src.kinematics_mujoco import MujocoKinematics
from src.kinematics_ur import URKinematics
from src.model_loader import load_robot_model
from src.observer import ArmObserver
from src.robot_backend_mujoco import MujocoBackend

STEP_DEG = 2.0
RATE = 0.02
TAU_S = 0.10            # first-order lag approximating servoJ lookahead
FINAL_DEG = 0.5
OVER_DEG = 1.0
QD_RMS = 0.10
PEAK_QD = 0.50


class DelayedBackend:
    # Delay the joint command and pass it through a first-order lag before the
    # MuJoCo position servo sees it.
    def __init__(self, inner, delay_s, tau_s, dt):
        self.inner = inner
        self.delay = max(0, int(round(float(delay_s) / dt)))
        self.tau = max(0.0, float(tau_s))
        self.dt = float(dt)
        self.queue = []
        self.hold = np.asarray(inner.joint_positions(), float).copy()
        self.q_f = self.hold.copy()
        self.grip = 0.0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def command(self, q, grip):
        self.queue.append(np.asarray(q, float).copy())
        if self.delay == 0:
            return
        self.grip = grip

    def step(self):
        if self.delay == 0:
            q_d = self.queue[-1] if self.queue else self.hold
            self.queue.clear()
        else:
            if len(self.queue) > self.delay:
                q_d = self.queue.pop(0)
            else:
                q_d = self.hold
            self.hold = np.asarray(q_d, float).copy()
        a = 1.0 if self.tau <= 0.0 else self.dt / (self.tau + self.dt)
        self.q_f = self.q_f + a * (np.asarray(q_d, float) - self.q_f)
        self.inner.command(self.q_f, self.grip)
        self.inner.step()


def run_case(model, data, dyn, delay_s, gain_scale, lqr_scale, seed):
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    q0 = np.asarray(data.qpos[:cfg.ARM_DOF], float).copy()

    inner = MujocoBackend(model, data, cfg.ARM_DOF)
    delayed = DelayedBackend(inner, delay_s, TAU_S, cfg.UR_CYCLE_DT)
    plant = MujocoKinematics(model, inner.tool_body_id, cfg.ARM_DOF)
    arm = Arm(model, data, kinematics=plant, backend=delayed)

    kp = cfg.UR_SERVO_KP * float(gain_scale)
    kv = cfg.UR_SERVO_KV * float(gain_scale)
    obs_kin = URKinematics(
        dyn, (np.full(cfg.ARM_DOF, kp), np.full(cfg.ARM_DOF, kv)),
        tool_pos=(0.0, 0.0, 0.0), tool_quat=(1.0, 0.0, 0.0, 0.0),
        root_yaw=cfg.UR_ROOT_YAW)
    obs = ArmObserver(model, data, q0, dt=cfg.UR_CYCLE_DT, kin=obs_kin,
                      lqr_weights=lqr_weights())
    arm.attach_observer(obs, LQRController(obs, gain_scale=lqr_scale),
                        Encoder(noisy=True, seed=seed))

    q_tgt = q0.copy()
    q_tgt[1] += np.radians(STEP_DEG)
    trace = []
    error = None
    n = int(2.5 / cfg.UR_CYCLE_DT)
    try:
        for _ in range(n):
            arm.command_step(q_tgt, cfg.GRIP_OPEN)
            trace.append((np.asarray(data.qpos[:cfg.ARM_DOF], float).copy(),
                          np.asarray(data.qvel[:cfg.ARM_DOF], float).copy(),
                          np.asarray(obs.xhat, float).copy()))
    except Exception as exc:
        error = '%s: %s' % (type(exc).__name__, exc)
    finally:
        arm.detach_observer()

    if not trace:
        return dict(delay_s=float(delay_s), gain_scale=float(gain_scale),
                    lqr_scale=float(lqr_scale), final_deg=float('nan'),
                    overshoot_deg=float('nan'), settle_s=float('nan'),
                    qdot_rms=float('nan'), peak_qd=float('nan'),
                    stable=False, error=error)

    q = np.array([t[0] for t in trace])
    qd = np.array([t[1] for t in trace])
    xh = np.array([t[2] for t in trace])
    err = q[:, 1] - q_tgt[1]
    dt = cfg.UR_CYCLE_DT
    sign = 1.0 if q_tgt[1] >= q0[1] else -1.0
    over = float(np.max(np.maximum(0.0, err * sign)))
    bad = np.nonzero(np.abs(err) > np.radians(0.10))[0]
    settle = float(bad[-1] * dt) if len(bad) else 0.0
    qd_err = xh[:, cfg.ARM_DOF:] - qd
    rms = float(np.sqrt(np.mean(qd_err * qd_err)))
    peak = float(np.max(np.abs(qd)))
    final = float(abs(err[-1]))
    stable = bool(error is None
                   and np.all(np.isfinite(q))
                   and np.degrees(final) <= FINAL_DEG
                   and np.degrees(over) <= OVER_DEG
                   and settle <= 1.5
                   and rms <= QD_RMS
                   and peak <= PEAK_QD)
    return dict(delay_s=float(delay_s), gain_scale=float(gain_scale),
                lqr_scale=float(lqr_scale),
                final_deg=float(np.degrees(final)),
                overshoot_deg=float(np.degrees(over)),
                settle_s=float(settle), qdot_rms=float(rms),
                peak_qd=float(peak), stable=stable, error=error)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(ROOT), 'notes', 'state_space_bench.json'))
    args = ap.parse_args()

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')
    data = mujoco.MjData(model)
    dyn = ArmDynamics.from_file(os.path.join(
        ROOT, 'models', 'ur3e', 'arm_dynamics.json'))

    cases = []
    print('%-8s %-6s %-6s %8s %9s %8s %8s %8s %s'
          % ('delay', 'gain', 'lqr', 'final', 'overshoot', 'settle',
             'qdotRMS', 'peakQD', 'stable'))
    for scale in (0.5, 1.0, 2.0):
        for delay in (0.0, 0.04, 0.08, 0.12):
            for lqr_scale in (0.25, 0.5, 1.0):
                r = run_case(model, data, dyn, delay, scale, lqr_scale,
                             seed=7)
                cases.append(r)
                print('%-8.0f %-6.1f %-6.2f %8.2f %9.2f %8.2f %8.3f %8.3f %s'
                      % (r['delay_s'] * 1e3, r['gain_scale'],
                         r['lqr_scale'], r['final_deg'],
                         r['overshoot_deg'], r['settle_s'], r['qdot_rms'],
                         r['peak_qd'], 'PASS' if r['stable'] else 'FAIL'))
    stable = sum(1 for r in cases if r['stable'])
    default = [r for r in cases
               if abs(r['lqr_scale'] - cfg.LQR_GAIN_SCALE) < 1e-9]
    default_ok = sum(1 for r in default if r['stable'])
    print('\nall gains: %d/%d pass the robustness envelope'
          % (stable, len(cases)))
    print('default LQR scale %.2f: %d/%d pass'
          % (cfg.LQR_GAIN_SCALE, default_ok, len(default)))
    with open(args.out, 'w') as fh:
        json.dump(dict(step_deg=STEP_DEG, rate=RATE, tau_s=TAU_S,
                       final_deg=FINAL_DEG, overshoot_deg=OVER_DEG,
                       qdot_rms=QD_RMS, peak_qd=PEAK_QD,
                       cases=cases), fh, indent=2)
    print('wrote %s' % args.out)
    return 0 if default_ok == len(default) else 1


if __name__ == '__main__':
    raise SystemExit(main())
