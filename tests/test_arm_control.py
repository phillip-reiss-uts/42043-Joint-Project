"""Hardware smoke test: does THIS machine drive the arm -- and is its config the
cell's, not the simulator's?

WHY A SEPARATE FILE FROM test_hardware.py
-----------------------------------------
test_hardware.py used to carry a --real mode that was a stub: it listed what
should be checked and recorded every item as SKIP, so it could not fail. That
mode has been deleted. This file is the opposite: a small number of checks
that really run against the controller and really fail.

The config check comes FIRST and is not advisory. Everything measured on this
cell -- the 180 deg root yaw, the table centre at -y, the true home -- lives in
src/config.py. That file is committed now, so a fresh clone carries it; the
silent failure left is a checkout of one of the OTHER branches (multi-object,
origin/simulation), whose config describes a different workcell. Every test
below would then pass against the wrong cell. So the values are asserted, not
printed.

WHAT IT DOES
------------
  1. config is the measured cell          (no motion)
  2. RTDE connects and returns joints     (no motion)
  3. one joint moves 2 deg at the paced 125 Hz rate   (MOTION, ~2 deg)

Run:
    python3 tests/test_arm_control.py --config-only
    python3 tests/test_arm_control.py

--config-only skips 2 and 3, so it is safe with the arm anywhere.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The cell as measured on 2026-10-23. These are not stylistic defaults: they are
# what the arm and the bench were found to be. See the comments in config.py.
EXPECT_IP = '192.168.0.194'
EXPECT_YAW = np.pi
EXPECT_TABLE_CENTRE = [0.0, -0.270]
EXPECT_TABLE_HALF = [0.22, 0.15]
# The work surface is the BASE PLANE -- the robot is bolted to the table, so the
# tabletop is z = 0. This used to assert 0.2315, which was never the table: it is
# the FLANGE-to-lowest-pad TOOL length, and it now lives with the tool geometry in
# config.py. Asserting it here failed the config gate on a correct tree.
EXPECT_TABLE_HEIGHT = 0.015
EXPECT_HOME0 = 1.5708          # shoulder_pan at the true home (was -0.366)

STEP_DEG = 2.0
CYCLES = 125                   # ~1 s at 125 Hz
STEP_TOL_MS = 2.5              # pacing tolerance on the calculated 8 ms


def check_config(cfg):
    print('=' * 70)
    print('1. config is the measured cell')
    print('=' * 70)
    bad = []

    def want(name, got, expect, tol=1e-6):
        # Strings (the IP) compare by equality; numbers by tolerance. Trying to
        # push both through np.allclose fails on the IP.
        if isinstance(got, str) or isinstance(expect, str):
            ok = str(got) == str(expect)
        else:
            ok = np.allclose(np.asarray(got, float),
                             np.asarray(expect, float), atol=tol)
        print('  %-16s %-28s %s' % (name, got, 'OK' if ok else
                                    'MISMATCH -- expected %s' % (expect,)))
        if not ok:
            bad.append(name)

    want('UR_IP', cfg.UR_IP, EXPECT_IP, tol=0)
    want('UR_ROOT_YAW', round(cfg.UR_ROOT_YAW, 6), round(EXPECT_YAW, 6),
         tol=1e-6)
    want('TABLE_CENTRE', list(cfg.TABLE_CENTRE), EXPECT_TABLE_CENTRE,
         tol=1e-6)
    want('TABLE_HALF', list(cfg.TABLE_HALF), EXPECT_TABLE_HALF, tol=1e-6)
    want('TABLE_HEIGHT', cfg.TABLE_HEIGHT, EXPECT_TABLE_HEIGHT, tol=1e-6)
    want('UR_HOME_Q[0]', round(cfg.UR_HOME_Q[0], 4), EXPECT_HOME0, tol=1e-4)

    if bad:
        print()
        print('  FAIL: %s do not match the measured cell.' % ', '.join(bad))
        print('  This tree does not describe the cell. Check the branch -- main is')
        print('  the cell, multi-object and origin/simulation are not -- and that')
        print('  src/config.py has not been edited locally. Every result below')
        print('  would otherwise describe a different workcell.')
        return False
    print('  all measured values present')
    return True


def check_rtde(cfg):
    print()
    print('=' * 70)
    print('2. RTDE connects')
    print('=' * 70)
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(cfg.UR_IP)
    try:
        q = np.array(r.getActualQ(), float)
        tcp = np.array(r.getActualTCPPose(), float)
    finally:
        r.disconnect()          # one control client at a time on a UR
    print('  q   :', np.round(np.degrees(q), 2))
    print('  TCP :', np.round(tcp[:3] * 1000, 1), 'mm (flange)')
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        print('  FAIL: did not get six finite joints')
        return False
    return True


def check_motion(cfg):
    print()
    print('=' * 70)
    print('3. one joint, %.1f deg, paced at %.0f Hz'
          % (STEP_DEG, 1.0 / cfg.UR_CYCLE_DT))
    print('=' * 70)
    print('  MOTION: clear the workcell, pendant in Remote, slider low,')
    print('  hand on the e-stop.')
    from src.robot_backend_ur import connect

    b = connect(cfg.UR_IP, activate_gripper=False)
    # The arm motion is what this tests, so the gripper is taken out of the loop
    # entirely. That satisfies the backend contract without exercising whichever
    # route the RG2 is on (onRobot XML-RPC primary, UR programs over the
    # dashboard as the fallback).
    b._gripper = None
    try:
        q0 = np.array(b.joint_positions(), float)
        print('  start :', np.round(np.degrees(q0), 2))

        tgt = q0.copy()
        tgt[0] += np.radians(STEP_DEG)
        b.command(tgt, 0.0)

        t0 = time.monotonic()
        for i in range(CYCLES):
            b.step()
            if i % 25 == 0:
                print('    t=%.2fs  q1=%6.2f deg'
                      % (time.monotonic() - t0,
                         np.degrees(b.joint_positions()[0])))
        elapsed = time.monotonic() - t0
        q1 = np.array(b.joint_positions(), float)
        print('  end   :', np.round(np.degrees(q1), 2))
    finally:
        b.close()

    step_ms = elapsed / CYCLES * 1000.0
    moved = np.degrees(q1[0] - q0[0])
    other = np.degrees(np.abs(q1[1:] - q0[1:])).max()
    print()
    print('  mean step : %.1f ms  (calculated %.1f)' % (step_ms,
                                                        cfg.UR_CYCLE_DT * 1000))
    print('  joint 1   : %+.2f deg' % moved)
    print('  others    : max %.2f deg (should be ~0)' % other)

    ok = True
    if abs(moved - STEP_DEG) > 0.5:
        print('  FAIL: expected %+.1f deg on joint 1' % STEP_DEG)
        ok = False
    if abs(step_ms - cfg.UR_CYCLE_DT * 1000) > STEP_TOL_MS:
        print('  FAIL: pacing off by more than %.1f ms' % STEP_TOL_MS)
        print('        servoJ does not block on this build, so step() sleeps')
        print('        the cycle out itself -- if this drifts, the machine')
        print('        cannot sustain the control rate.')
        ok = False
    if other > 1.0:
        print('  FAIL: a joint other than 1 moved -- joint order or coupling')
        ok = False
    if ok:
        print('  PASS: motion, direction, distance and pacing')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Smoke-test this machine against the real UR3e.')
    ap.add_argument('--config-only', action='store_true',
                    help='only check src/config.py; no robot contact')
    args = ap.parse_args(argv)

    from src import config as cfg
    print('machine: %s' % sys.executable)
    print()

    if not check_config(cfg):
        return 1
    if args.config_only:
        print()
        print('  (--config-only: no robot contact attempted)')
        return 0

    if not check_rtde(cfg):
        return 1
    if not check_motion(cfg):
        return 1

    print()
    print('=' * 70)
    print('  this machine drives the arm, on the cell\'s own config')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
