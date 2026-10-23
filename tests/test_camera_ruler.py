# Checklist Test 5: the camera ruler check -- does a detected cube land where
# it physically is, compared with a tape measure?
#
# WHY THIS IS THE HARD ONE
# ------------------------
# Every other camera test is self-consistent: the deprojection round-trips, the
# intrinsics are sane, the table filter separates object from surface. All of
# that can pass while the estimate is 30 mm out, because every one of those
# checks uses the same camera model to make its claim. Only a measurement taken
# with a ruler and a pair of hands catches that, and this script exists to
# record exactly those numbers.
#
# WHAT TO DO
# ----------
#   1. Put the cube at a spot you have measured, in the ROBOT BASE FRAME.
#      The easy way is to jog the closed gripper to the cube first and read
#      getActualTCPPose(); that gives you (x, y) without a tape measure.
#   2. Run this with --truth x,y in METRES, relative to the base.
#      Do it for the centre of the table, a near corner and a far corner.
#   3. Compare the printed error. The checklist expects 3-5 mm at the centre
#      and worse at the corners.
#
# The tool frame is the one thing that will be wrong first: CAM_MOUNT_POS/QUAT
# in src/config.py are still the SIMULATOR's bracket placement, so a bias here
# is expected until they are measured. A constant bias across all three spots
# is the mount; an error that GROWS across the table is the extrinsics or a
# scale mistake.
#
# Run:
#     python tests/test_camera_ruler.py --truth 0.0,-0.27
#     python tests/test_camera_ruler.py --truth 0.15,-0.15 --replay frame.npz
#
# --replay reads a frame saved by tools/record_real_frame.py, so the analysis
# can be repeated on the laptop with no camera attached.

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _camera_from_flange():
    # gripper base -> camera optical frame, from the configured bracket.
    from src.arm_dynamics import rot_from_quat
    from src import config as cfg
    T = np.eye(4)
    T[:3, :3] = rot_from_quat(cfg.CAM_MOUNT_QUAT)
    T[:3, 3] = cfg.CAM_MOUNT_POS
    return T


def _world(depth_source, kin, backend, cfg):
    # Exactly the wiring run_hardware.py uses, so the test exercises the same
    # path the pick does rather than a convenient one.
    from src.world_real import RealObjectWorld, FixedCameraWorld, \
        camera_transform
    flange_pose = lambda: kin.pose(backend.joint_positions())
    if cfg.CAMERA_MODE == 'fixed':
        pos = np.array(cfg.FIXED_CAM_POS, float)
        tgt = np.array(cfg.FIXED_CAM_TARGET, float)
        T = camera_transform(pos, tgt - pos, cfg.FIXED_CAM_UP)
        return FixedCameraWorld(depth_source, flange_pose=flange_pose,
                                T_base_cam=T)
    return RealObjectWorld(depth_source, flange_pose=flange_pose,
                           T_flange_cam=_camera_from_flange())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Test 5: compare a detected cube with a measured one.')
    ap.add_argument('--truth', default=None,
                    help='measured cube xy in the base frame, metres: x,y')
    ap.add_argument('--replay', default=None,
                    help='a .npz from tools/record_real_frame.py, instead of '
                         'a live camera')
    ap.add_argument('--no-robot', action='store_true',
                    help='with --replay: describe the camera pose yourself '
                         'rather than reading it from the arm')
    ap.add_argument('--cam-pos', default=None,
                    help='with --no-robot: camera position x,y,z (metres)')
    ap.add_argument('--cam-target', default=None,
                    help='with --no-robot: aim point x,y,z (metres)')
    args = ap.parse_args(argv)

    from src import config as cfg
    from src.arm_dynamics import ArmDynamics
    from src.kinematics_ur import URKinematics

    dyn = ArmDynamics.from_file(os.path.join(ROOT, 'models', 'ur3e',
                                             'arm_dynamics.json'))

    # ---- depth source ----------------------------------------------------
    # There is no replay source in this repo yet (the docstring in
    # world_real.py mentions tools/record_real_frame.py, which does not exist),
    # so a live camera is the only path. A saved .npz could be loaded by a
    # small DepthSource here later.
    if args.replay:
        raise SystemExit('--replay needs a replay DepthSource; none exists in '
                         'the repo yet. Use a live camera, or --no-robot with '
                         'a saved frame once one is recorded.')
    from src.world_real import RealSenseSource
    depth = RealSenseSource()
    print('[camera] live D435i')

    # ---- where the camera is --------------------------------------------
    if args.no_robot:
        if not (args.cam_pos and args.cam_target):
            raise SystemExit('--no-robot needs --cam-pos and --cam-target')
        from src.world_real import camera_transform
        pos = np.array([float(v) for v in args.cam_pos.split(',')])
        tgt = np.array([float(v) for v in args.cam_target.split(',')])
        T_world_cam = camera_transform(pos, tgt - pos)
        world = _StaticCameraWorld(depth, T_world_cam)
        print('[camera] static pose from the command line')
    else:
        from src.robot_backend_ur import connect
        backend = connect(cfg.UR_IP, activate_gripper=False)
        backend._gripper = None
        kin = URKinematics(dyn, (np.full(6, cfg.UR_SERVO_KP),
                                 np.full(6, cfg.UR_SERVO_KV)),
                           tool_pos=cfg.TOOL_MOUNT_POS,
                           tool_quat=cfg.TOOL_MOUNT_QUAT,
                           root_yaw=cfg.UR_ROOT_YAW)
        world = _world(depth, kin, backend, cfg)
        p, _ = kin.pose(backend.joint_positions())
        print('[camera] %s, tool at %s' % (cfg.CAMERA_MODE, np.round(p, 3)))

    # ---- detect ----------------------------------------------------------
    views = world.perceive()
    depth.close()

    ok = [v for v in views if v.ok]
    print()
    print('detected %d object(s), %d usable' % (len(views), len(ok)))
    if not ok:
        print('  Nothing detected. Checklist Test 5 failure modes:')
        print('    * outside the view, or the object is too small/large for')
        print('      _span_ok\'s size band')
        print('    * depth units wrong -- see tests/test_camera_enum.py step 2')
        print('    * the cube is not within SURFACE_BAND (%.1f mm) of '
              'TABLE_HEIGHT (%.3f)'
              % (cfg.SURFACE_BAND * 1000, cfg.TABLE_HEIGHT))
        return 1

    if args.truth is None:
        print()
        for i, v in enumerate(ok):
            print('  #%d  xy = (%.4f, %.4f) m  yaw %6.1f deg  n=%d'
                  % (i, v.xy[0], v.xy[1], v.yaw, len(getattr(v, 'points', []))))
        print()
        print('Re-run with --truth x,y (metres, base frame) to get the error.')
        return 0

    truth = np.array([float(s) for s in args.truth.split(',')])
    best = max(ok, key=lambda v: getattr(v, 'n_points', 0))
    err = np.asarray(best.xy, float) - truth
    print()
    print('  truth    : (%+.4f, %+.4f) m' % (truth[0], truth[1]))
    print('  detected : (%+.4f, %+.4f) m' % (best.xy[0], best.xy[1]))
    print('  error    : (%+.1f, %+.1f) mm   |e| = %.1f mm'
          % (err[0] * 1000, err[1] * 1000, np.linalg.norm(err) * 1000))
    print()
    print('  Checklist: 3-5 mm at the centre, worse at a corner.')
    print('  A CONSTANT bias across all three spots is CAM_MOUNT; an error')
    print('  that grows across the table is the extrinsics or a scale error.')
    return 0


class _StaticCameraWorld:
    # ObjectWorld with the camera pose given directly, for --no-robot runs.
    # Subclasses DepthWorld so the deprojection, table filter and detection are
    # the same code the robot runs -- only camera_pose() differs.

    def __init__(self, depth_source, T_world_cam):
        from src.world_real import DepthWorld

        class _W(DepthWorld):
            def camera_pose(inner):
                return T[:3, 3].copy(), T[:3, :3].copy()

        T = np.asarray(T_world_cam, float)
        self._w = _W(depth_source, lambda: (np.zeros(3), np.eye(3)))

    def camera_pose(self):
        return self._w.camera_pose()

    def perceive(self):
        return self._w.perceive()

    def gripper_pose(self):
        return np.zeros(3), np.eye(3)

    def close(self):
        self._w.close()


if __name__ == '__main__':
    raise SystemExit(main())
