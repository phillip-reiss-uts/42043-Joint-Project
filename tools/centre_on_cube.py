#!/usr/bin/env python3
"""Centre the gripper over the cube as perception sees it, then stop.

No grasp, no descent, no gripper. One question: does the arm go where the
camera believes the cube is? Run it, then look at the cube.

    python tools/centre_on_cube.py --dry-run     # print the target only
    python tools/centre_on_cube.py --looks 3     # move, look, correct twice

Each look prints the cube position it believes. After the first move the next
look should report the same xy; if it keeps sliding as the arm moves, the mount
is wrong and the arm is centring on a point that moves with the camera.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg
from src.arm import Arm
from src.arm_dynamics import ArmDynamics, rot_from_quat
from src.kinematics_ur import URKinematics
from src.robot_backend_ur import connect
from src.world_real import RealObjectWorld, RealSenseSource


def mount():
    # FLANGE -> camera. CAM_MOUNT_* is flange-relative, so the kinematics must
    # report the flange frame for this to compose correctly.
    T = np.eye(4)
    T[:3, :3] = rot_from_quat(cfg.CAM_MOUNT_QUAT)
    T[:3, 3] = np.array(cfg.CAM_MOUNT_POS, float)
    return T


def main():
    ap = argparse.ArgumentParser(
        description='Hold the gripper centred over the perceived cube.')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the target and do not move')
    ap.add_argument('--looks', type=int, default=3,
                    help='perceive/move cycles (default 3)')
    ap.add_argument('--tries', type=int, default=30,
                    help='looks to wait for the cube before giving up')
    ap.add_argument('--ip', default=cfg.UR_IP)
    args = ap.parse_args()

    dyn = ArmDynamics.from_file(os.path.join(ROOT, 'models', 'ur3e',
                                             'arm_dynamics.json'))
    # root_yaw matters HERE, unlike the calibration tools: the cell frame is
    # rotated onto the table, and without it the deprojected cloud lands tens of
    # millimetres outside TABLE_XY_BOUNDS, so every look returns nothing.
    kin = URKinematics(dyn, (np.full(6, cfg.UR_SERVO_KP),
                             np.full(6, cfg.UR_SERVO_KV)),
                       tool_pos=(0.0, 0.0, 0.0),
                       tool_quat=(1.0, 0.0, 0.0, 0.0),
                       root_yaw=cfg.UR_ROOT_YAW)
    backend = connect(args.ip, activate_gripper=False, dt=cfg.UR_CYCLE_DT,
                      lookahead_time=cfg.UR_LOOKAHEAD_TIME,
                      gain=cfg.UR_SERVO_GAIN)
    backend._gripper = None
    arm = Arm(kinematics=kin, backend=backend)

    depth = RealSenseSource()
    world = RealObjectWorld(
        depth,
        flange_pose=lambda: kin.pose(
            np.asarray(backend.joint_positions(), float)),
        T_flange_cam=mount())
    # the same hover the servo uses: PICK_Z lifted by the standoff, i.e. the
    # camera CAM_HOVER_HEIGHT above the table.
    z_hover = cfg.PICK_Z + cfg.SERVO_STANDOFF

    try:
        for i in range(args.looks):
            # Wait for a look with the cube in it: the routine is usually
            # started from wherever the arm was left, which is not a camera
            # pose. A miss is not a failure, it is a "move and look again".
            views = []
            for attempt in range(args.tries):
                views = [v for v in world.perceive() if v.ok]
                if views:
                    break
                if attempt == 0:
                    print('  nothing seen (point count is on the [PERC] line '
                          'above)')
                input('  move the arm until the cube is in view, then Enter ')
            if not views:
                print('no cube after %d looks -- giving up' % args.tries)
                return 2
            v = views[0]
            p = np.array([v.centre[0], v.centre[1], z_hover])
            # max_joint is the UR3e's own range (+-2*pi). Arm.solve_ik defaults
            # to +-3 rad, which is a simulator convention: a wound-up wrist
            # needs past it, the solver clips, and the arm then drives to a pose
            # that is NOT the target and stops well short of it.
            q = arm.solve_ik(p, np.asarray(backend.joint_positions(), float),
                             R_target=cfg.R_DOWN, max_joint=2.0 * np.pi)
            p_ik, _ = kin.pose(q[:cfg.ARM_DOF])
            short = float(np.linalg.norm(p_ik - p))
            if short > 2.0e-3:
                print('  IK stopped %.0f mm short of the target -- not moving'
                      % (short * 1e3))
                print('  (the solver found no pose within the joint range)')
                return 2
            print('look %d  cube (%+.3f, %+.3f)  target (%+.3f, %+.3f, %.3f)'
                  % (i + 1, v.centre[0], v.centre[1], p[0], p[1], p[2]))
            if args.dry_run:
                break
            arm.move(q, cfg.GRIP_OPEN, rate=cfg.FEED)

            p_now, _ = kin.pose(np.asarray(backend.joint_positions(), float))
            print('  moved: tool at (%+.3f, %+.3f, %.3f), %.0f mm from target'
                  % (p_now[0], p_now[1], p_now[2],
                     1e3 * float(np.linalg.norm(p_now - p))))

        p_tool, _ = kin.pose(np.asarray(backend.joint_positions(), float))
        print('tool now at (%+.3f, %+.3f, %.3f)' % tuple(p_tool))
    finally:
        depth.close()
        backend.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
