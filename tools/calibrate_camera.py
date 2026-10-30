#!/usr/bin/env python3
"""Camera extrinsic calibration. Three methods, two kinds of output:

    --touch / --aruco   a FIXED (eye-to-hand) camera
                        -> FIXED_CAM_POS / TARGET / UP
    --wrist             a WRIST (eye-in-hand) camera, printed marker
                        -> CAM_POS / CAM_QUAT   (CAM_MOUNT_* alias them)
    --wrist-touch       a WRIST camera, no marker: the cube IS the target
                        -> CAM_POS / CAM_QUAT

    python tools/calibrate_camera.py --touch
    python tools/calibrate_camera.py --aruco
    python tools/calibrate_camera.py --wrist
    python tools/calibrate_camera.py --wrist-touch

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

--wrist-touch
          The ruler method without the marker. At each station the pads are
          lowered onto the cube's top, which fixes that cube's centre in the
          base frame; lift so the cube is in view and capture, which fixes the
          same point in the camera frame. The mount is the rigid transform
          between the two point sets (Kabsch), so no marker, no PnP and no
          touch-to-board conversion are involved. The camera height IS
          observable here, unlike the fixed --touch. Twist the wrist between
          stations for the same reason as --wrist.

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
    # root_yaw is required for anything that runs the PERCEPTION: the table
    # filters (TABLE_XY_BOUNDS, SURFACE_Z_MIN) live in the cell frame, and
    # without it the deprojected cloud lands in the mirrored quadrant and every
    # look is rejected -- which is why --wrist-touch never saw the cube. The
    # marker fits cancel the base rotation, so this changes nothing for --wrist.
    return URKinematics(dyn,
                        (np.full(6, cfg.UR_SERVO_KP), np.full(6, cfg.UR_SERVO_KV)),
                        tool_pos=cfg.TOOL_MOUNT_POS,
                        tool_quat=cfg.TOOL_MOUNT_QUAT,
                        root_yaw=cfg.UR_ROOT_YAW)


def _base_xy(receive, kin):
    # The CONTACT POINT, not the gripper base.
    #
    # The operator touches the marker with the pads, so the FK pose (gripper
    # base) has to be moved down the tool axis by base->pad before it is a point
    # on the board. Using the base directly put the board GRIPPER_TO_PAD_MM above
    # the table, which shifts every camera estimate and makes the error grow as
    # the wrist turns -- the "spread 132 mm / 117 deg" failure, not a flexing
    # mount. The tool axis is +z, pointing down the approach.
    q = receive.getActualQ()
    p, R = kin.pose(np.asarray(q, float))
    return np.asarray(p, float) + np.asarray(R, float)[:, 2] * (
        cfg.GRIPPER_TO_PAD_MM / 1000.0)


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
    print('  Touch corner 0 with the CLOSED pads\' midpoint over it, then Enter.')
    input('  > ')
    a = _base_xy(receive, kin)
    print('  Touch corner 1 -- the adjacent one -- then Enter.')
    input('  > ')
    b = _base_xy(receive, kin)

    print('  contact z %.1f mm (expect ~0 with the pads on the table)'
          % (a[2] * 1e3))
    if abs(a[2]) > 0.02 or abs(b[2]) > 0.02:
        print('[CAL] the touch was not made on the table: the pads sit '
              '%.1f / %.1f mm above it. The edge length cannot detect this, '
              'so the board would be placed high and every pose would shift.'
              % (a[2] * 1e3, b[2] * 1e3))

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
    samples = []      # (T_tool, T_base_cam) per pose; the board hypothesis is
                      # applied after the loop, so the four yaws can be tried.
    try:
        # ---- locate the board once -------------------------------------
        while True:
            line = input('  Press Enter to capture the marker   [q] quit > ')
            if line.strip().lower() == 'q':
                return 2
            colour, K = grab()
            found = aruco_corners(colour, args.dict)
            if found:
                break
            _check_frame(colour, K, found, args, 'calib_marker')
            print('[CAL] no marker detected -- check lighting, focus and the '
                  'dictionary name. Nothing was captured; try again.')
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
        print('  [v] saves a snapshot and reports the marker WITHOUT recording')
        print('      a pose -- use it to judge the view before committing.')
        print('  [d] stops early. A pose that finds no marker is not counted.')
        print()
        recorded = 0
        while recorded < args.poses:
            line = input('  pose %d/%d -- [Enter] record  [v] snapshot  '
                         '[x] undo last  [d] done > '
                         % (recorded + 1, args.poses)).strip().lower()
            if line == 'd':
                break
            if line == 'x':
                if samples:
                    samples.pop()
                    recorded -= 1
                    print('    last pose discarded (%d recorded).' % recorded)
                else:
                    print('    nothing recorded yet.')
                continue
            if line == 'v':
                colour, K = grab()
                _check_frame(colour, K, aruco_corners(colour, args.dict),
                             args, 'calib_pose_%02d' % (recorded + 1))
                continue

            colour, K = grab()
            found = aruco_corners(colour, args.dict)
            if not found:
                _check_frame(colour, K, found, args,
                             'calib_nomarker_%02d' % (recorded + 1))
                print('    [CAL] no marker -- pose NOT recorded. Move, then '
                      'record again, or [v] to inspect first.')
                continue

            try:
                T_tool = _tool_pose(receive, kin)
                # Seed PnP from where FK says the camera should be, converted
                # into the solver's own frame. A planar target has a mirror
                # solution, so a good seed is what keeps it physically right.
                T_init_cv = cam_mjocam_to_cv(T_tool @ X_nom)
                T_cam_cv, err = solve_aruco(colour, K, Tb, args.marker_size,
                                            dict_name=args.dict,
                                            T_init=T_init_cv)
            except RuntimeError as exc:
                print('    [CAL] %s -- pose NOT recorded' % exc)
                continue
            X = wrist_mount(T_tool, cam_cv_to_mjocam(T_cam_cv))
            r_tool = float(np.linalg.norm(X[:3, 3]))
            if err > 2.0 or not (0.02 <= r_tool <= 0.40):
                print('    [CAL] rejected: %.2f px, %.0f mm from the tool -- '
                      'poor fit, or not a wrist-sized mount'
                      % (err, r_tool * 1e3))
                continue
            samples.append((T_tool, T_cam_cv))
            recorded += 1
            print('    %.2f px   centre %.1f mm from the tool, '
                  '%.1f mm from the config value   (%d/%d)'
                  % (err, float(np.linalg.norm(X[:3, 3])) * 1e3,
                     float(np.linalg.norm(X[:3, 3] - X_nom[:3, 3])) * 1e3,
                     recorded, args.poses))
            if recorded >= 3:
                # Live error bar, so a pose that disagrees is caught while the
                # arm is still there -- [x] undoes it and it is retaken.
                try:
                    dev_t, dev_r, deg = _board_fits(Tb, samples)[0][:3]
                    print('    running spread: %.2f mm / %.2f deg '
                          '(board yaw %d deg, %d poses)'
                          % (dev_t * 1e3, dev_r, deg, recorded))
                except Exception:
                    pass
    finally:
        pipe.stop()

    if len(samples) < 3:
        print('[CAL] need at least 3 usable poses, got %d' % len(samples))
        return 2

    # ---- which edge of the marker did the touches follow ---------------
    # The touches fix the board's ORIGIN and one edge direction, but the
    # operator can pick the wrong adjacent corner. 0 -> 3 instead of 0 -> 1
    # leaves the board right-handed and on the table, merely rotated 90 deg,
    # and one rotated board wrecks every pose. So try the four yaws and keep
    # the one that makes the poses agree -- the data decides, not the guess.
    print()
    print('  board yaw check (touched edge vs marker +x):')
    rows = _board_fits(Tb, samples)
    best_deg = rows[0][2]
    for dev_t_i, dev_r_i, deg_i, _, _ in sorted(rows, key=lambda r: r[2]):
        print('    %3d deg : spread %8.2f mm / %7.2f deg%s'
              % (deg_i, dev_t_i * 1e3, dev_r_i,
                 '   <-- best' if deg_i == best_deg else ''))
    dev_t, dev_r, deg, X, Xs_best = rows[0]
    if deg:
        print('  using the %d deg board: the touched edge was not the marker'
              % deg)
        print('  +x edge. Next time touch corner 0 and then corner 1.')
    if dev_t > 5e-3 or dev_r > 2.0:
        print('  no board yaw explains the disagreement, so the touch points')
        print('  themselves are wrong -- wrong corner pair or origin, the')
        print('  marker moved between touches, or a pose with the marker')
        print('  oblique. Re-run the touches before the poses.')

    _report_mount(X, dev_t, dev_r, len(samples), X_nom, args.json)
    if dev_t > 5e-3 or dev_r > 2.0:
        _pose_diagnostics(X, Xs_best, dev_t)
    return 0 if (dev_t < 5e-3 and dev_r < 2.0) else 1


def run_wrist_touch(args):
    # Eye-in-hand from KNOWN POINTS -- the ruler method without the marker.
    #
    # Each station gives the SAME physical point twice: the cube's top centre,
    # once in the base frame (the pads touch it) and once in the camera frame
    # (the depth look sees it). The mount is the rigid transform between the two
    # point sets, so there is no board, no PnP, no corner order and no touch
    # conversion -- the failure modes that ate the marker runs are all absent.
    #
    # The camera HEIGHT is observable here: the base-frame point is a real 3D
    # point at TABLE + OBJECT_HEIGHT, not a plane with an assumed height.
    try:
        from src.world_real import RealSenseSource, RealObjectWorld
    except ImportError as exc:
        print('[CAL] %s' % exc)
        return 2

    receive = _connect(args.ip)
    kin = _kinematics()
    X_nom = _mount_from_config()

    depth = RealSenseSource()
    flange_pose = lambda: kin.pose(np.asarray(receive.getActualQ(), float))
    world = RealObjectWorld(depth, flange_pose=flange_pose,
                            T_flange_cam=X_nom)
    z_top = cfg.TABLE_HEIGHT + cfg.OBJECT_HEIGHT

    print()
    print('WRIST CALIBRATION  (robot as ruler, no marker)')
    print('  %d stations, spread over the table. At each one:' % args.stations)
    print('    1. lay the cube down and lower the CLOSED pads until they')
    print('       just touch its top (the same touch the board mode uses)')
    print('    2. lift until the cube is in view, TWIST THE WRIST, capture')
    print()

    Ps, Qs = [], []    # camera-frame point, same point in the tool frame
    try:
        i = 0
        while i < args.stations:
            input('  station %d/%d -- pads on the cube top, then Enter > '
                  % (i + 1, args.stations))
            q_touch = np.asarray(receive.getActualQ(), float)
            p, R = kin.pose(q_touch)
            contact = np.asarray(p, float) + np.asarray(R, float)[:, 2] * (
                cfg.GRIPPER_TO_PAD_MM / 1000.0)
            if abs(contact[2] - z_top) > 0.015:
                print('    contact z %.1f mm, cube top is %.1f mm -- lower the '
                      'pads until they just touch it'
                      % (contact[2] * 1e3, z_top * 1e3))
            P_true = np.array([contact[0], contact[1], z_top])

            input('  lift, put the cube in view, then Enter > ')
            views = [v for v in world.perceive() if v.ok]
            if not views:
                print('    no cube in view -- station skipped, try again')
                continue
            v = min(views, key=lambda w: float(np.linalg.norm(
                np.asarray(w.centre)[:2] - P_true[:2])))
            if v.hull is None:
                print('    no top face in this view -- station skipped')
                continue

            # The detector's xy is the top-face centre in the base frame; its
            # z comes from the hull, so this point is exactly the camera-frame
            # point carried through the NOMINAL mount. Using TABLE + H here
            # instead would inject the nominal mount's height error.
            p_det = np.array([v.centre[0], v.centre[1],
                              float(np.mean(v.hull[:, 2]))])
            T_tool = _pose_at(kin, np.asarray(receive.getActualQ(), float))
            T_cam = T_tool @ X_nom
            p_cam = (np.linalg.inv(T_cam) @ np.r_[p_det, 1.0])[:3]
            q_tool = (np.linalg.inv(T_tool) @ np.r_[P_true, 1.0])[:3]
            Ps.append(p_cam)
            Qs.append(q_tool)
            i += 1
            print('    touched (%+.3f, %+.3f), detected (%+.3f, %+.3f)'
                  % (P_true[0], P_true[1], p_det[0], p_det[1]))
    finally:
        depth.close()

    if len(Ps) < 4:
        print('[CAL] need at least 4 stations, got %d' % len(Ps))
        return 2

    P = np.array(Ps)
    Q = np.array(Qs)
    X = _kabsch(P, Q)
    res = np.linalg.norm(P @ X[:3, :3].T + X[:3, 3] - Q, axis=1)

    print()
    print('  per-station residual (the fit against the touched points):')
    for i, r in enumerate(res):
        print('    station %2d : %6.2f mm' % (i + 1, r * 1e3))

    # Rotation error bar: refit without each station and see how far the
    # rotation moves. Translation residuals alone cannot see a rotation that
    # the station layout fails to constrain.
    dev_r = 0.0
    if len(P) > 4:
        for i in range(len(P)):
            X_i = _kabsch(np.delete(P, i, 0), np.delete(Q, i, 0))
            c = float(np.clip((np.trace(X_i[:3, :3].T @ X[:3, :3]) - 1.0) / 2.0,
                              -1.0, 1.0))
            dev_r = max(dev_r, float(np.degrees(np.arccos(c))))
    dev_t = float(np.max(res))

    _report_mount(X, dev_t, dev_r, len(P), X_nom, args.json)
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
    print('  simulator and the hardware together. Leave')
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


def _annotate(colour, found, name='aruco_corners.png'):
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
    path = os.path.join(ROOT, name)
    cv2.imwrite(path, img)
    return path


def _check_frame(colour, K, found, args, tag):
    # Snapshot preview: write the annotated frame and say whether it is usable.
    # Records NOTHING -- this is the [v] key, so a pose can be judged before it
    # is committed. On this cell the only way to see the camera is a saved
    # image; pressing Enter blind is how a pose ends up oblique or half out of
    # frame, which is exactly what makes the poses disagree.
    path = _annotate(colour, found, name='%s.png' % tag)
    h, w = colour.shape[:2]
    if not found:
        print('    no marker in view -- image saved: %s' % path)
        return path

    fx = float(K[0, 0])
    for mid, pix in found:
        edges = [np.linalg.norm(pix[(i + 1) % 4] - pix[i]) for i in range(4)]
        edge_px = float(np.mean(edges))
        rng = fx * float(args.marker_size) / max(edge_px, 1e-9)
        margin = float(min(pix[:, 0].min(), pix[:, 1].min(),
                           w - 1 - pix[:, 0].max(), h - 1 - pix[:, 1].max()))
        warn = '   <-- close to the frame edge' if margin < 0.05 * min(w, h) else ''
        print('    id %d: %.0f px edge, ~%.0f mm away, %.0f px from the edge%s'
              % (mid, edge_px, rng * 1e3, margin, warn))
    print('    %d marker(s) detected -- image saved: %s' % (len(found), path))
    return path


def _kabsch(P, Q):
    # Rigid X with Q ~ X P, from matched 3D points. SVD, no scipy, and it
    # cannot return a reflection (D keeps det +1) -- a mirrored clamp is not a
    # thing, so a reflected fit would be a bug wearing a number.
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    U, _s, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = qc - R @ pc
    return X


def _pose_at(kin, q):
    p, R = kin.pose(np.asarray(q, float))
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    return T


def _board_fits(Tb, samples):
    # One X set per board-yaw hypothesis, best spread first:
    # (dev_t, dev_r, deg, X, Xs). Used by the live running spread as well as
    # the final table, so both always name the same board.
    inv_Tb = np.linalg.inv(Tb)
    rows = []
    for deg in (0, 90, 180, 270):
        Tb_try = Tb @ _rot_z(deg)
        Xs = [wrist_mount(T_tool, cam_cv_to_mjocam(Tb_try @ inv_Tb @ T_cam))
              for T_tool, T_cam in samples]
        X, dev_t, dev_r = mean_transform(Xs)
        rows.append((dev_t, dev_r, deg, X, Xs))
    rows.sort(key=lambda r: r[0])
    return rows


def _pose_diagnostics(X, Xs, dev_t):
    # Which pose is dragging the mean. One bad capture is the common case: the
    # marker was oblique or near the frame edge, PnP still fitted it, and the
    # mean moved. Naming the pose -- and the spread without it -- turns the
    # warning into a decision.
    rows = []
    for i, X_i in enumerate(Xs):
        dt = float(np.linalg.norm(X_i[:3, 3] - X[:3, 3]))
        c = float(np.clip((np.trace(X[:3, :3].T @ X_i[:3, :3]) - 1.0) / 2.0,
                          -1.0, 1.0))
        rows.append((dt, float(np.degrees(np.arccos(c))), i))

    print()
    print('  per-pose deviation from the mean:')
    for dt, dr, i in rows:
        flag = '   <-- outlier' if dt > max(2.0 * dev_t, 5e-3) else ''
        print('    pose %2d : %7.1f mm / %5.2f deg%s'
              % (i + 1, dt * 1e3, dr, flag))

    if len(Xs) > 4:
        best = None
        for i in range(len(Xs)):
            _, dt_i, dr_i = mean_transform(Xs[:i] + Xs[i + 1:])
            if best is None or dt_i < best[1]:
                best = (i, dt_i, dr_i)
        print('    dropping pose %d would leave %.2f mm / %.2f deg'
              % (best[0] + 1, best[1] * 1e3, best[2]))


def _rot_z(deg):
    # 4x4 rotation about +z, for trying the four board yaws.
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return T


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
    g.add_argument('--wrist-touch', action='store_true',
                   help='wrist camera from touched cube positions, no marker')
    ap.add_argument('--ip', default=cfg.UR_IP)
    ap.add_argument('--marker-size', type=float, default=0.10,
                    help='marker edge length in metres (aruco/wrist). The '
                         'default is the shipped 100 mm marker; measure and '
                         'pass the real edge if you print another size')
    ap.add_argument('--dict', default='DICT_4X4_50',
                    help='ArUco dictionary name (aruco/wrist)')
    ap.add_argument('--poses', type=int, default=12,
                    help='arm poses to average over (wrist; at least 3)')
    ap.add_argument('--stations', type=int, default=6,
                    help='cube positions to touch (wrist-touch; at least 4)')
    ap.add_argument('--json', default=None,
                    help='also write the result to this file')
    args = ap.parse_args(argv)

    print('=' * 72)
    print('CAMERA EXTRINSICS  (%s camera)'
          % ('wrist' if (args.wrist or args.wrist_touch) else 'fixed'))
    print('=' * 72)
    print('  robot  : %s (receive only; hand-guide the arm)' % args.ip)
    print('  table  : TABLE_HEIGHT = %.4f m -- the fit trusts this'
          % cfg.TABLE_HEIGHT)
    try:
        if args.wrist:
            return run_wrist(args)
        if args.wrist_touch:
            return run_wrist_touch(args)
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
