#!/usr/bin/env python3
"""Camera extrinsic calibration. Three methods, two kinds of output:

    --touch / --aruco   a FIXED (eye-to-hand) camera
                        -> FIXED_CAM_POS / TARGET / UP
    --wrist             a WRIST (eye-in-hand) camera
                        -> CAM_POS / CAM_QUAT   (CAM_MOUNT_* alias them)

    python tools/calibrate_camera.py --touch
    python tools/calibrate_camera.py --aruco
    python tools/calibrate_camera.py --wrist

The marker size defaults to 0.10 m because that is the marker in models/aruco/.
Pass --marker-size with the MEASURED edge if you print a different size: a
wrong value here scales the whole fit, and it fails like a bad mount.

--touch   The robot is the ruler. Place the cube, centre the closed gripper
          over it, lower until it just touches the top, press Enter. Repeat at
          six or more positions spread over the table. No OpenCV, no target.

          The camera HEIGHT is not observable this way (see solve_touch), so it
          is taken from the current config value. Measure it first, or use
          --aruco.

--aruco   Print a marker, lay it flat on the table, touch two adjacent corners
          with the TCP, then capture. solvePnP from one image. Needs OpenCV
          (`opencv-contrib-python`) and observes all six degrees of freedom.

--wrist   The same marker and the same two touches locate the board ONCE, in
          the base frame. Then the arm is moved to several poses and the board
          is photographed from each. Every pose yields one estimate of the same
          constant -- tool -> camera -- so the SPREAD across poses is the error
          bar, not just a number to average away. This is the eye-in-hand case:
          the camera moves, the board does not, and no AX = XB factorisation is
          needed because the board's base-frame pose is known from the touches.

          Move mostly by TWISTING THE WRIST between poses. Translation-only
          motion barely constrains the mount's rotation, which is the part a
          printed clamp gets wrong.

Only the RTDE *receive* interface is used: the arm can be hand-guided in local
mode, and nothing here commands motion. The pendant stays in charge.
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

from src import config as cfg
from src.calibration_camera import (
    board_pose_from_two_touches, cam_cv_to_mjocam, cam_mjocam_to_cv,
    mean_transform, quat_from_rot, solve_aruco, solve_touch,
    transform_to_pos_target, wrist_mount,
)
from src.world_real import camera_transform


def _config_transform():
    # The current config values as a 4x4, used as the solver's starting point.
    pos = np.array(cfg.FIXED_CAM_POS, float)
    target = np.array(cfg.FIXED_CAM_TARGET, float)
    return camera_transform(pos, target - pos, cfg.FIXED_CAM_UP)


def _mount_from_config():
    # gripper base -> camera, as config currently has it. The solver's seed and
    # the reference every sanity check is measured against.
    from src.arm_dynamics import rot_from_quat
    T = np.eye(4)
    T[:3, :3] = rot_from_quat(cfg.CAM_MOUNT_QUAT)
    T[:3, 3] = np.array(cfg.CAM_MOUNT_POS, float)
    return T


def _connect(ip):
    # Receive-only: no control interface, so the pendant keeps authority and
    # the arm can be jogged by hand.
    from rtde_receive import RTDEReceiveInterface
    return RTDEReceiveInterface(ip)


def _kinematics():
    from src.arm_dynamics import ArmDynamics
    from src.kinematics_ur import URKinematics
    path = os.path.join(ROOT, 'models', 'ur3e', 'arm_dynamics.json')
    dyn = ArmDynamics.from_file(path)
    return URKinematics(dyn,
                        (np.full(6, cfg.UR_SERVO_KP), np.full(6, cfg.UR_SERVO_KV)),
                        tool_pos=cfg.TOOL_MOUNT_POS,
                        tool_quat=cfg.TOOL_MOUNT_QUAT)


def _base_xy(receive, kin):
    q = receive.getActualQ()
    p, _ = kin.pose(np.asarray(q, float))
    return p


def _tool_pose(receive, kin):
    # Full pose of the controlled frame (gripper base). The eye-in-hand product
    # is inv(this) @ T_base_cam, so the rotation matters as much as the origin.
    q = receive.getActualQ()
    p, R = kin.pose(np.asarray(q, float))
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    return T


def _board_from_touches(receive, kin, args):
    # Two adjacent marker corners, touched with the TCP, fix the board's pose in
    # the base frame. Shared by both marker modes so the operator is asked for
    # the same thing either way, and so the diagonal-vs-edge mistake is caught
    # in one place.
    print('  Touch corner 0 with the TCP (pads\' midpoint over the corner), '
          'then Enter.')
    input('  > ')
    a = _base_xy(receive, kin)
    print('  Touch corner 1 -- the adjacent one -- then Enter.')
    input('  > ')
    b = _base_xy(receive, kin)

    edge = float(np.linalg.norm(b - a))
    print('  touched edge %.1f mm (marker is %.1f mm)'
          % (edge * 1e3, args.marker_size * 1e3))
    if abs(edge - args.marker_size) > 0.15 * args.marker_size:
        if abs(edge - args.marker_size * np.sqrt(2.0)) < 0.15 * args.marker_size:
            print('[CAL] that is the DIAGONAL (%.1f mm). Touch two adjacent '
                  'corners, 0 and 1.' % (args.marker_size * np.sqrt(2.0) * 1e3))
        else:
            print('[CAL] the touched edge does not match --marker-size. '
                  'Measure the printed edge and pass the real value.')
        return None
    return board_pose_from_two_touches(a, b)


def _report(T, rms, method, out_json=None):
    pos, target, up = transform_to_pos_target(T)
    print()
    print('=' * 72)
    print('RESULT  (%s)  rms %.2f mm' % (method, rms * 1e3))
    print('=' * 72)
    print('FIXED_CAM_POS    = (%.5f, %.5f, %.5f)' % tuple(pos))
    print('FIXED_CAM_TARGET = (%.5f, %.5f, %.5f)' % tuple(target))
    print('FIXED_CAM_UP     = (%.5f, %.5f, %.5f)' % tuple(up))
    print()
    print('Paste those into src/config.py, set CAMERA_MODE = \'fixed\', and')
    print('re-run the ruler check: place the cube at three measured spots and')
    print('compare the detected centre. A wrong fit shows as a bias that grows')
    print('across the table.')
    if out_json:
        with open(out_json, 'w') as fh:
            json.dump({'T_base_cam': T.tolist(),
                       'FIXED_CAM_POS': list(map(float, pos)),
                       'FIXED_CAM_TARGET': list(map(float, target)),
                       'FIXED_CAM_UP': list(map(float, up)),
                       'rms_m': float(rms)}, fh, indent=2)
        print('wrote %s' % out_json)


def run_touch(args):
    from src.world_real import RealSenseSource

    receive = _connect(args.ip)
    kin = _kinematics()
    src = RealSenseSource()

    print()
    print('TOUCH CALIBRATION')
    print('  For each station: place the cube, centre the CLOSED gripper over')
    print('  it, lower until the pads just touch the top face.')
    print('  Spread the stations over the table: near, far, left, right.')
    print('  At least 4; 6-8 gives a better fit. The camera height is taken')
    print('  from the current config (%.3f m).' % cfg.FIXED_CAM_POS[2])

    frames, targets = [], []
    try:
        while True:
            line = input('[Enter] capture  [d] done  (%d so far) > '
                         % len(frames)).strip().lower()
            if line == 'd':
                break
            depth, K = src.capture()
            p = _base_xy(receive, kin)
            frames.append((depth, K))
            targets.append((float(p[0]), float(p[1])))
            print('  captured at x %.4f  y %.4f' % (p[0], p[1]))
    finally:
        src.close()

    if len(frames) < 4:
        print('[CAL] need at least 4 stations, got %d' % len(frames))
        return 2

    T0 = _config_transform()
    T, rms, n = solve_touch(frames, targets, T0)
    _report(T, rms, 'touch, %d stations' % n, args.json)
    return 0 if rms < 0.010 else 1


def run_aruco(args):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print('[CAL] pyrealsense2 is not installed')
        return 2

    receive = _connect(args.ip)
    kin = _kinematics()

    print()
    print('ARUCO CALIBRATION')
    print('  Marker: %s, edge %.1f mm' % (args.dict, args.marker_size * 1e3))
    print('  Lay it FLAT on the table, matte side up, no tape wrinkles.')
    print()

    # Capture FIRST, then touch. The board must not move between the image and
    # the touches that locate it.
    pipe = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(conf)
    align = rs.align(rs.stream.color)
    try:
        input('  Press Enter to capture the marker > ')
        fs = align.process(pipe.wait_for_frames())
        colour = np.asanyarray(fs.get_color_frame().get_data())
        i = fs.get_color_frame().profile.as_video_stream_profile() \
            .get_intrinsics()
        K = np.array([[i.fx, 0.0, i.ppx],
                      [0.0, i.fy, i.ppy],
                      [0.0, 0.0, 1.0]], float)
    finally:
        pipe.stop()

    from src.calibration_camera import aruco_corners
    found = aruco_corners(colour, args.dict)
    if not found:
        print('[CAL] no marker detected -- check lighting, focus and the '
              'dictionary name')
        return 2

    # cv2's corner order encodes the marker's own orientation: the printed bits
    # break the square's 90-degree symmetry, so corner 0 is a specific physical
    # corner. Show the operator which.
    path = _annotate(colour, found)
    print('  detected %d marker(s); annotated image: %s'
          % (len(found), path))
    print('  Open it and find the corners labelled 0 and 1.')
    print()

    Tb = _board_from_touches(receive, kin, args)
    if Tb is None:
        return 2

    T_cv, err = solve_aruco(colour, K, Tb, args.marker_size,
                            dict_name=args.dict, T_init=_config_transform())
    # solvePnP works in the OpenCV camera frame (y down, z forward); everything
    # that consumes these numbers -- camera_transform, world_points_from_depth,
    # FixedCameraWorld -- is in the MuJoCo one (y up, z backward). Converting
    # here is the difference between a camera looking at the table and one
    # looking at the ceiling. See CAM_CV_TO_MJ in src/calibration_camera.py.
    T = cam_cv_to_mjocam(T_cv)
    # Convert the reprojection error to metres at the board's distance, so the
    # touch and ArUco modes report the same kind of number.
    dist = float(np.linalg.norm(T[:3, 3] - Tb[:3, 3]))
    rms_m = err * dist / max(abs(K[0, 0]), 1e-9)
    _report(T, rms_m, 'aruco, %.2f px reprojection' % err, args.json)
    return 0 if err < 2.0 else 1


def run_wrist(args):
    # Eye-in-hand. The board lies still on the table; the camera rides the arm.
    #
    # Each pose gives T_base_cam straight from PnP and T_base_tool straight from
    # FK, so inv(T_base_tool) T_base_cam is an estimate of the ONE constant that
    # describes the mount. Averaging is the easy half; the SPREAD is the useful
    # half, because it is the only measurement of "the clamp does not move".
    #
    # No AX = XB factorisation: that machinery exists for when the target's pose
    # in the base frame is unknown, and the two corner touches already fix it.
    try:
        import pyrealsense2 as rs
    except ImportError:
        print('[CAL] pyrealsense2 is not installed')
        return 2

    receive = _connect(args.ip)
    kin = _kinematics()

    print()
    print('WRIST CALIBRATION  (eye-in-hand)')
    print('  Marker: %s, edge %.1f mm' % (args.dict, args.marker_size * 1e3))
    print('  Lay it FLAT on the table, matte side up, no tape wrinkles, and do')
    print('  not move it again -- every pose is measured against it.')
    print()

    pipe = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(conf)

    def grab():
        fs = pipe.wait_for_frames()
        colour = np.asanyarray(fs.get_color_frame().get_data())
        i = fs.get_color_frame().profile.as_video_stream_profile() \
            .get_intrinsics()
        K = np.array([[i.fx, 0.0, i.ppx],
                      [0.0, i.fy, i.ppy],
                      [0.0, 0.0, 1.0]], float)
        return colour, K

    from src.calibration_camera import aruco_corners
    Xs = []
    try:
        # ---- locate the board once -------------------------------------
        input('  Press Enter to capture the marker > ')
        colour, K = grab()
        found = aruco_corners(colour, args.dict)
        if not found:
            print('[CAL] no marker detected -- check lighting, focus and the '
                  'dictionary name')
            return 2
        path = _annotate(colour, found)
        print('  detected %d marker(s); annotated image: %s'
              % (len(found), path))
        print('  Open it and find the corners labelled 0 and 1.')
        print()
        Tb = _board_from_touches(receive, kin, args)
        if Tb is None:
            return 2

        # ---- one estimate per arm pose ---------------------------------
        X_nom = _mount_from_config()
        print()
        print('  Now move the arm and photograph the marker from %d poses.'
              % args.poses)
        print('  TWIST THE WRIST between poses, and keep the marker in view.')
        print('  Enter alone repeats a pose; [d] stops early.')
        print()
        for k in range(args.poses):
            line = input('  pose %d/%d -- move, then Enter  [d] done > '
                         % (k + 1, args.poses)).strip().lower()
            if line == 'd':
                break
            try:
                colour, K = grab()
                found = aruco_corners(colour, args.dict)
                if not found:
                    raise RuntimeError('no marker in this frame')
                T_tool = _tool_pose(receive, kin)
                # Seed PnP from where FK says the camera should be, converted
                # into the solver's own frame. A planar target has a mirror
                # solution, so a good seed is what keeps it physically right.
                T_init_cv = cam_mjocam_to_cv(T_tool @ X_nom)
                T_cam_cv, err = solve_aruco(colour, K, Tb, args.marker_size,
                                            dict_name=args.dict,
                                            T_init=T_init_cv)
            except RuntimeError as exc:
                print('    [CAL] %s -- skipped' % exc)
                continue
            X = wrist_mount(T_tool, cam_cv_to_mjocam(T_cam_cv))
            Xs.append(X)
            print('    %.2f px   centre %.1f mm from the tool, '
                  '%.1f mm from the config value'
                  % (err, float(np.linalg.norm(X[:3, 3])) * 1e3,
                     float(np.linalg.norm(X[:3, 3] - X_nom[:3, 3])) * 1e3))
    finally:
        pipe.stop()

    if len(Xs) < 3:
        print('[CAL] need at least 3 usable poses, got %d' % len(Xs))
        return 2

    X, dev_t, dev_r = mean_transform(Xs)
    _report_mount(X, dev_t, dev_r, len(Xs), X_nom, args.json)
    return 0 if (dev_t < 5e-3 and dev_r < 2.0) else 1


def _report_mount(X, dev_t, dev_r, n, X_nom, out_json=None):
    # Print the mount in the form config stores, with its error bar.
    pos = X[:3, 3]
    quat = quat_from_rot(X[:3, :3])
    r = float(np.linalg.norm(pos))

    print()
    print('=' * 72)
    print('RESULT  (wrist, %d poses)  spread %.2f mm / %.2f deg'
          % (n, dev_t * 1e3, dev_r))
    print('=' * 72)
    print('CAM_POS  = (%.5f, %.5f, %.5f)' % tuple(pos))
    print('CAM_QUAT = (%.6f, %.6f, %.6f, %.6f)     # w, x, y, z' % tuple(quat))
    print()
    print('  CAM_MOUNT_POS/_QUAT alias these two, so this one edit moves the')
    print('  simulator and the hardware together (todo F7). Leave')
    print('  HW_CAM_DELTA_* at identity: this IS the measurement it exists for.')
    print()
    print('  camera centre is %.1f mm from the tool frame' % (r * 1e3))
    print('  moved from the config value by %.1f mm' % (np.linalg.norm(
        pos - X_nom[:3, 3]) * 1e3))
    if not (0.040 <= r <= 0.100):
        print('  WARNING: outside the 40-100 mm the bracket should give. A')
        print('           translation in the hundreds of mm, or a spread far')
        print('           above a couple of mm, is a frame or units error.')
    if dev_t > 5e-3 or dev_r > 2.0:
        print('  WARNING: the poses disagree by more than 5 mm / 2 deg, so one')
        print('           constant does not describe them all. Suspicion order:')
        print('           the mount flexes, the board moved, or a pose was')
        print('           captured with the marker badly oblique or half out')
        print('           of frame. Re-run before trusting these numbers.')
    print()
    print('  Then check it: put the cube at three measured spots and compare')
    print('  the detected centre against a ruler (tests/test_camera_ruler.py).')
    if out_json:
        with open(out_json, 'w') as fh:
            json.dump({'T_tool_cam': X.tolist(),
                       'CAM_POS': list(map(float, pos)),
                       'CAM_QUAT_wxyz': list(map(float, quat)),
                       'spread_m': float(dev_t),
                       'spread_deg': float(dev_r),
                       'poses': int(n)}, fh, indent=2)
        print('  wrote %s' % out_json)


def _annotate(colour, found):
    # Save the capture with each detected corner numbered, so the operator can
    # see which physical corner is 0 and which is 1.
    import cv2
    img = colour.copy()
    for mid, pix in found:
        p = np.asarray(pix, float).round().astype(int)
        cv2.polylines(img, [p], True, (0, 255, 0), 2)
        for k in range(4):
            cv2.circle(img, tuple(p[k]), 7, (0, 0, 255), -1)
            cv2.putText(img, str(k), (p[k][0] + 10, p[k][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(img, 'id %d' % mid, (p[0][0] + 10, p[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    path = os.path.join(ROOT, 'aruco_corners.png')
    cv2.imwrite(path, img)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='calibrate_camera.py',
        description='Camera extrinsics: fixed (eye-to-hand) or wrist '
                    '(eye-in-hand).')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--touch', action='store_true',
                   help='robot touches the cube at several stations (fixed)')
    g.add_argument('--aruco', action='store_true',
                   help='printed marker, solvePnP (needs OpenCV; fixed)')
    g.add_argument('--wrist', action='store_true',
                   help='printed marker from several arm poses (wrist)')
    ap.add_argument('--ip', default=cfg.UR_IP)
    ap.add_argument('--marker-size', type=float, default=0.10,
                    help='marker edge length in metres (aruco/wrist). The '
                         'default is the shipped 100 mm marker; measure and '
                         'pass the real edge if you print another size')
    ap.add_argument('--dict', default='DICT_4X4_50',
                    help='ArUco dictionary name (aruco/wrist)')
    ap.add_argument('--poses', type=int, default=12,
                    help='arm poses to average over (wrist; at least 3)')
    ap.add_argument('--json', default=None,
                    help='also write the result to this file')
    args = ap.parse_args(argv)

    print('=' * 72)
    print('CAMERA EXTRINSICS  (%s camera)'
          % ('wrist' if args.wrist else 'fixed'))
    print('=' * 72)
    print('  robot  : %s (receive only; hand-guide the arm)' % args.ip)
    print('  table  : TABLE_HEIGHT = %.4f m -- the fit trusts this'
          % cfg.TABLE_HEIGHT)
    try:
        if args.wrist:
            return run_wrist(args)
        return run_aruco(args) if args.aruco else run_touch(args)
    except ImportError as exc:
        print('[CAL] missing dependency: %s' % exc)
        print('      marker modes need opencv-contrib-python and '
              'pyrealsense2; touches need ur_rtde')
        return 2
    except KeyboardInterrupt:
        print('\n[CAL] cancelled')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
