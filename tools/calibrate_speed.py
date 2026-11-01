#!/usr/bin/env python3
"""Calibrate a safe hardware servo speed scale.

The arm runs one short straight-line move at a sequence of software speeds.
Every candidate is bounded independently of the value under test:

    cumulative command <= CAL_V_MAX
    joint velocity     <= CAL_JOINT_MAX
    path               <= CAL_MOVE_M + margin
    following error    <= CAL_FOLLOW_M
    elapsed time       <= expected * 2

Any breach stops the run and latches. Nothing is retried automatically.

The scale is relative to SERVO_V_TRANSIT. The tool only writes
UR_SPEED_SCALE when --write is given.

TWO STAGES. Run --stage low with the pendant at 10% first. That proves the
safety checks before the arm is asked for speed at 100%. Only then run
--stage full at pendant 100%; those numbers are the ones to save.

    python tools/calibrate_speed.py --stage low
    python tools/calibrate_speed.py --stage full --write
"""

import argparse
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
from src.kinematics_ur import URKinematics
from src.robot_backend_ur import connect
from src.servo import PBVSServo, _tool_yaw
from src.world import TargetView

CONFIG_PATH = os.path.join(ROOT, 'src', 'config.py')

# Independent safety envelope. These do not move with the candidate scale.
CAL_V_MAX = 0.020                # m/s, absolute
CAL_JOINT_MAX = 0.30             # rad/s, any joint
CAL_FOLLOW_M = 0.005             # m, final error
CAL_OVERSHOOT_M = 0.003          # m
CAL_MOVE_M = 0.060               # m, test path
CAL_BOX_M = 0.080                # m, allowed excursion from start
CAL_START_CLEAR_M = 0.15         # m above table

CANDIDATE_SPEEDS = (0.004, 0.006, 0.008, 0.012, 0.016, 0.020)


class FixedTargetWorld:
    # The servo needs a world. This one reports the live flange pose and a
    # fixed target; no camera, no cube.
    def __init__(self, arm, target, yaw):
        self.arm = arm
        self.target = np.asarray(target, float)
        self.yaw = float(yaw)

    def gripper_pose(self):
        return self.arm.kin.pose(
            np.asarray(self.arm.backend.joint_positions(), float))

    def perceive(self):
        return [TargetView(centre=self.target.copy(), yaw=self.yaw,
                           span=np.array([0.06, 0.06], float),
                           score=1.0, n_points=6000)]


def _pose(kin, backend):
    return kin.pose(np.asarray(backend.joint_positions(), float))


def _hold(backend, q, seconds=0.30):
    for _ in range(max(1, int(seconds / backend.dt))):
        backend.command(q, cfg.GRIP_OPEN)
        backend.step()


def _inside_table(xy):
    cx, cy, hx, hy = cfg.TABLE_XY_BOUNDS
    return (abs(float(xy[0]) - cx) <= hx
            and abs(float(xy[1]) - cy) <= hy)


def _check_start(p0, distance, axis):
    if not _inside_table(p0[:2]):
        raise RuntimeError('start point is outside TABLE_XY_BOUNDS; move the '
                           'tool over the table first')
    if p0[2] < cfg.TABLE_HEIGHT + CAL_START_CLEAR_M:
        raise RuntimeError('tool is only %.0f mm above the table; move it '
                           'to a clear hover first'
                           % ((p0[2] - cfg.TABLE_HEIGHT) * 1e3))
    target = np.array(p0, float)
    target[axis] += distance
    if not _inside_table(target[:2]):
        target = np.array(p0, float)
        target[axis] -= distance
    if not _inside_table(target[:2]):
        raise RuntimeError('both test directions leave TABLE_XY_BOUNDS')
    return target


def _write_scale(scale):
    src = open(CONFIG_PATH).read()
    line = 'UR_SPEED_SCALE = %.4f' % scale
    new, n = re.subn(r'^UR_SPEED_SCALE = .*$', line, src, count=1,
                     flags=re.MULTILINE)
    if n != 1:
        print('  could not find UR_SPEED_SCALE in config -- NOT written')
        return False
    open(CONFIG_PATH, 'w').write(new)
    print('  wrote %s' % CONFIG_PATH)
    return True


def _run_one(backend, arm, kin, p0, q0, target, speed, scale):
    # One candidate. Returns (passed, detail).
    world = FixedTargetWorld(arm, target, _tool_yaw(_pose(kin, backend)[1]))
    sv = PBVSServo(
        None, None, arm, world,
        speed_scale=scale,
        use_kf=False,
        perceive_every=10 ** 9,
        v_max=cfg.SERVO_V_TRANSIT,
        v_descent=cfg.SERVO_V_DESCENT,
        tol_p=0.004, tol_o=0.05,
        dq_max=0.02)
    sv.p_est = target.copy()
    sv.yaw_est = _tool_yaw(_pose(kin, backend)[1])
    sv.best_yaw = sv.yaw_est
    sv.best_p = target.copy()
    sv.best_span = 0.06

    t0 = time.monotonic()
    last_t = [t0]
    last_p = [p0.copy()]
    peak_v = [0.0]
    peak_qd = [0.0]
    reason = [None]

    def aborted():
        now = time.monotonic()
        dt = max(now - last_t[0], 1e-6)
        last_t[0] = now
        p, _ = _pose(kin, backend)
        v = float(np.linalg.norm(p - last_p[0]) / dt)
        last_p[0] = p.copy()
        qd = np.asarray(backend.joint_velocities(), float)
        qdv = float(np.max(np.abs(qd))) if len(qd) else 0.0
        peak_v[0] = max(peak_v[0], v)
        peak_qd[0] = max(peak_qd[0], qdv)
        actual_limit = min(CAL_V_MAX,
                           max(1.5 * speed, speed + 0.003))
        if v > actual_limit:
            reason[0] = ('actual speed %.3f m/s (requested %.3f, limit %.3f)'
                         % (v, speed, actual_limit))
        elif qdv > CAL_JOINT_MAX:
            reason[0] = 'joint speed %.2f rad/s' % qdv
        elif np.linalg.norm(p - p0) > CAL_BOX_M:
            reason[0] = 'left the %.0f mm test box' % (CAL_BOX_M * 1e3)
        elif now - t0 > max(5.0, 3.0 * CAL_MOVE_M / max(speed, 1e-6) + 2.0):
            reason[0] = 'timeout'
        return reason[0] is not None

    max_iters = max(2000, int(3.0 * CAL_MOVE_M / max(speed, 1e-6)
                               / backend.dt))
    ok, _, _ = sv.servo(target[2], max_iters=max_iters, aborted=aborted)
    p_end, _ = _pose(kin, backend)
    final = float(np.linalg.norm(p_end - target))
    d = target - p0
    length = float(np.linalg.norm(d))
    along = float(np.dot(p_end - p0, d / max(length, 1e-9)))
    over = max(0.0, along - length)

    passed = (reason[0] is None and ok and final <= CAL_FOLLOW_M
              and over <= CAL_OVERSHOOT_M
              and peak_qd[0] <= CAL_JOINT_MAX)
    detail = dict(speed_m_s=speed, scale=scale, passed=passed,
                  aborted=bool(reason[0]), reason=reason[0],
                  final_error_m=final, overshoot_m=over,
                  peak_speed_m_s=peak_v[0], peak_joint_rad_s=peak_qd[0])
    return passed, detail


def main():
    ap = argparse.ArgumentParser(
        description='Calibrate a safe software servo speed scale.')
    ap.add_argument('--ip', default=cfg.UR_IP)
    ap.add_argument('--stage', choices=('low', 'full'), default='low')
    ap.add_argument('--axis', choices=('x', 'y'), default='x')
    ap.add_argument('--distance', type=float, default=CAL_MOVE_M)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    start_run_log(sys.argv[1:], ROOT, tag='calibrate_speed')

    if args.stage == 'low':
        print('STAGE 1: pendant at 10%. This proves the safety checks.')
    else:
        print('STAGE 2: pendant at 100%. The saved scale comes from here.')
    print('Cell clear, no cube, hand on the pendant stop.')
    input('Press Enter to confirm > ')

    dyn = ArmDynamics.from_file(os.path.join(ROOT, 'models', 'ur3e',
                                             'arm_dynamics.json'))
    kin = URKinematics(dyn, (np.full(cfg.ARM_DOF, cfg.UR_SERVO_KP),
                             np.full(cfg.ARM_DOF, cfg.UR_SERVO_KV)),
                       tool_pos=(0.0, 0.0, 0.0),
                       tool_quat=(1.0, 0.0, 0.0, 0.0),
                       root_yaw=cfg.UR_ROOT_YAW)
    backend = connect(args.ip, activate_gripper=False,
                      dt=cfg.UR_CYCLE_DT,
                      lookahead_time=cfg.UR_LOOKAHEAD_TIME,
                      gain=cfg.UR_SERVO_GAIN)
    arm = Arm(kinematics=kin, backend=backend)

    best = None
    try:
        p0, _ = _pose(kin, backend)
        q0 = np.asarray(backend.joint_positions(), float).copy()
        axis = 0 if args.axis == 'x' else 1
        target = _check_start(p0, float(args.distance), axis)
        q_goal = arm.solve_ik(target, q0, R_target=_pose(kin, backend)[1],
                              max_joint=2.0 * np.pi)
        p_ik, _ = kin.pose(q_goal[:cfg.ARM_DOF])
        if np.linalg.norm(p_ik - target) > 2e-3:
            raise RuntimeError('test target is not reachable by IK')
        print('start  (%+.4f, %+.4f, %.4f)' % tuple(p0))
        print('target (%+.4f, %+.4f, %.4f)  distance %.0f mm'
              % (target[0], target[1], target[2],
                 np.linalg.norm(target - p0) * 1e3))

        candidates = CANDIDATE_SPEEDS
        if args.stage == 'low':
            candidates = candidates[:1]
        for speed in candidates:
            if speed > CAL_V_MAX:
                break
            scale = speed / cfg.SERVO_V_TRANSIT
            if scale > 1.0:
                break
            arm.move(q0, cfg.GRIP_OPEN, rate=0.03)
            _hold(backend, q0)
            print('  candidate %.3f m/s  scale %.3f' % (speed, scale))
            passed, detail = _run_one(backend, arm, kin, p0, q0, target,
                                      speed, scale)
            print('    %s  final %.1f mm  over %.1f mm  peak v %.3f  '
                  'peak qd %.2f%s'
                  % ('PASS' if passed else 'FAIL',
                     detail['final_error_m'] * 1e3,
                     detail['overshoot_m'] * 1e3,
                     detail['peak_speed_m_s'], detail['peak_joint_rad_s'],
                     '' if not detail['reason'] else '  -- ' + detail['reason']))
            if not passed:
                print('  stopping here; no faster candidate will be tried.')
                break
            best = detail

        if best is None:
            print('no passing candidate; UR_SPEED_SCALE not changed')
            return 2
        print('recommended UR_SPEED_SCALE = %.4f  (%.3f m/s)'
              % (best['scale'], best['speed_m_s']))
        if args.write:
            answer = input('write this to src/config.py? [y/N] > ')
            if answer.strip().lower().startswith('y'):
                _write_scale(best['scale'])
    finally:
        try:
            arm.move(q0, cfg.GRIP_OPEN, rate=0.02)
        except Exception:
            pass
        backend.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
