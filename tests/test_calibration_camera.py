# Camera extrinsic calibration: offline tests, no robot and no camera.
#
# The two solvers in src/calibration_camera.py are pure maths, so they can be
# checked against a synthetic camera whose pose is known. That is the only way
# to know a calibration tool is right before pointing it at a real cell.
#
# Run: python tests/test_calibration_camera.py

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.calibration_camera import (
    board_pose_from_two_touches, pnp_pose, R_to_rotvec, rotvec_to_R,
    solve_touch, transform_from, transform_to_pos_target,
)
from src.world_real import camera_transform


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))
    return bool(ok)


def main():
    failures = 0
    print()
    print('=' * 74)
    print('CAMERA EXTRINSICS  (offline solvers)')
    print('=' * 74)

    # ---------------- rotation helpers ----------------
    print('\nROTATION')
    rv = np.array([0.3, -0.7, 0.2])
    R = rotvec_to_R(rv)
    failures += not check('rotvec -> R -> rotvec round-trips',
                          np.allclose(R_to_rotvec(R), rv, atol=1e-12))
    R_pi = rotvec_to_R(np.array([np.pi - 1e-7, 0.0, 0.0]))
    failures += not check('the near-pi case is handled',
                          np.allclose(R_to_rotvec(R_pi), [np.pi - 1e-7, 0, 0],
                                      atol=1e-6))

    # ---------------- config round trip ----------------
    print('\nCONFIG ROUND TRIP')
    T = transform_from([0.15, -0.42, 0.83], [0.2, -0.1, 0.05])
    pos, target, up = transform_to_pos_target(T, table_z=0.02)
    T2 = camera_transform(pos, target - pos, up)
    failures += not check(
        'pos/target/up reproduce the pose through camera_transform',
        np.allclose(T2, T, atol=1e-9),
        'max diff %.2e' % np.max(np.abs(T2 - T)))
    failures += not check(
        'the target lies on the table plane',
        abs(target[2] - 0.02) < 1e-9, 'target z %.4f' % target[2])

    # ---------------- touch solver ----------------
    # Synthetic camera: a known pose, a handful of 3D points in the camera
    # frame, and a detector that projects them through the candidate pose.
    print('\nTOUCH SOLVER')
    T_true = transform_from([0.10, -0.40, 0.78], [0.10, -0.08, 0.03])
    Xc = [np.array(v, float) for v in (
        (-0.10, -0.05, 0.70), (0.05, -0.08, 0.66), (0.14, 0.02, 0.72),
        (-0.12, 0.09, 0.68), (0.02, 0.12, 0.75), (0.11, -0.11, 0.69),
    )]

    def detect(T, depth, K):
        # `depth` carries the frame index.
        p = T @ np.r_[Xc[int(depth)], 1.0]
        return p[:2]

    targets = [(T_true @ np.r_[p, 1.0])[:2] for p in Xc]
    frames = [(i, None) for i in range(len(Xc))]
    # The solver holds z fixed (xy residuals cannot observe camera height),
    # so the initial guess must carry the measured height.
    T0 = transform_from([0.13, -0.37, 0.78], [0.12, -0.05, 0.01])

    T_fit, rms, n = solve_touch(frames, targets, T0, detect=detect)
    failures += not check(
        'recovers a known camera pose from touch correspondences',
        np.allclose(T_fit, T_true, atol=1e-7),
        'max diff %.2e m, rms %.2e m' % (np.max(np.abs(T_fit - T_true)), rms))
    failures += not check('reports the station count', n == len(Xc))

    # ---------------- aruco board frame ----------------
    print('\nARUCO BOARD FRAME')
    a = np.array([0.20, 0.00, 0.022])
    b = np.array([0.26, 0.00, 0.022])
    Tb = board_pose_from_two_touches(a, b)
    Rb = Tb[:3, :3]
    failures += not check(
        'the board frame is orthonormal with det +1',
        np.allclose(Rb @ Rb.T, np.eye(3), atol=1e-12)
        and abs(np.linalg.det(Rb) - 1.0) < 1e-12)
    failures += not check(
        'its x axis runs from the first touched corner to the second',
        np.allclose(Rb[:, 0], [1.0, 0.0, 0.0], atol=1e-12))
    failures += not check(
        'its origin is the first touched corner',
        np.allclose(Tb[:3, 3], a, atol=1e-12))

    # ---------------- PnP convention (needs OpenCV) ----------------
    print('\nPNP CONVENTION')
    try:
        import cv2
    except ImportError:
        print('  [SKIP] OpenCV not installed -- ArUco mode untested here')
    else:
        s = 0.05
        obj = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]], float)
        obj_base = (Rb @ obj.T).T + Tb[:3, 3]
        # A camera that sees the board: build its pose, then project.
        T_cam = transform_from([0.05, -0.55, 0.75], [0.08, -0.04, 0.02])
        R_cb = T_cam[:3, :3].T
        t_cb = -R_cb @ T_cam[:3, 3]
        K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
        pix, _ = cv2.projectPoints(obj_base, cv2.Rodrigues(R_cb)[0], t_cb,
                                   K, None)
        # A planar target has a mirror solution, so seed with a perturbed
        # estimate of the camera pose -- exactly what the tool has from config.
        T_init = transform_from([0.07, -0.52, 0.77], [0.10, -0.02, 0.04])
        recovered, err = pnp_pose(obj_base, pix, K, T_init=T_init)
        failures += not check(
            'solvePnP + inversion returns the camera pose, not its inverse',
            np.allclose(recovered, T_cam, atol=1e-6),
            'max diff %.2e, reprojection %.2e px'
            % (np.max(np.abs(recovered - T_cam)), err))

    print()
    print('=' * 74)
    if failures:
        print('  %d check(s) FAILED' % failures)
        print('=' * 74)
        return 1
    print('  Solvers recover a known pose; conventions round-trip.')
    print('  Still needs a real cell: the fit is only as good as the touches')
    print('  and the measured TABLE_HEIGHT.')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
