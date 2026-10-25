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
    board_pose_from_two_touches, cam_cv_to_mjocam, cam_mjocam_to_cv,
    CAM_CV_TO_MJ, mean_transform, pnp_pose, quat_from_rot, R_to_rotvec,
    rotvec_to_R, solve_touch, transform_from, transform_to_pos_target,
    wrist_mount,
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

    # ---------------- camera frame conventions ----------------
    print('\nCONVENTIONS')
    T_any = transform_from([0.12, -0.31, 0.64], [0.3, 0.1, -0.2])
    failures += not check(
        'cv <-> mujoco camera conversion is its own inverse',
        np.array_equal(cam_mjocam_to_cv(cam_cv_to_mjocam(T_any)), T_any))
    failures += not check(
        'it is a proper rotation (det +1)',
        abs(float(np.linalg.det(CAM_CV_TO_MJ)) - 1.0) < 1e-12,
        'det %+.3f' % np.linalg.det(CAM_CV_TO_MJ))
    failures += not check(
        'it flips exactly y and z, leaving the optical axis on -z',
        np.allclose(CAM_CV_TO_MJ, np.diag([1.0, -1.0, -1.0])))

    # ---------------- eye-in-hand algebra ----------------
    # A downward-looking wrist camera: the MuJoCo camera looks along its own -z,
    # so the identity rotation already points it at the table. Yaw and a small
    # tilt give the pose diversity the real procedure asks the operator for.
    print('\nWRIST MOUNT')
    X_true = transform_from([0.05, 0.0, 0.09], [0.0, 0.0, np.pi])
    tools = []
    for k in range(5):
        T = np.eye(4)
        T[:3, :3] = rotvec_to_R(np.array([0.0, 0.0, 0.35 * k])) \
            @ rotvec_to_R(np.array([0.12 * k, 0.0, 0.0]))
        T[:3, 3] = [0.02 * k, -0.55 + 0.02 * k, 0.75]
        tools.append(T)

    est = [wrist_mount(Tt, Tt @ X_true) for Tt in tools]
    failures += not check(
        'inv(T_tool) T_cam recovers a known mount from every pose',
        all(np.allclose(e, X_true, atol=1e-12) for e in est))

    T_base = transform_from([0.0, 0.0, 0.0], [0.0, 0.0, np.pi])
    failures += not check(
        'rotating the base frame does not change the mount',
        np.allclose(wrist_mount(T_base @ tools[0],
                                T_base @ tools[0] @ X_true), X_true,
                    atol=1e-12),
        'UR_ROOT_YAW cancels in the product')

    rng = np.random.default_rng(0)
    noisy = [T @ transform_from(rng.normal(0, 1e-4, 3),
                                rng.normal(0, 1e-3, 3)) for T in est]
    X_mean, dev_t, dev_r = mean_transform(noisy)
    failures += not check(
        'the mean sits inside the noise it was given',
        np.linalg.norm(X_mean[:3, 3] - X_true[:3, 3]) < 1e-3 and dev_r < 0.5,
        'off %.2f mm, spread %.2f mm / %.2f deg'
        % (np.linalg.norm(X_mean[:3, 3] - X_true[:3, 3]) * 1e3,
           dev_t * 1e3, dev_r))
    failures += not check(
        'the quaternion is wxyz and pastes into config',
        np.allclose(quat_from_rot(X_true[:3, :3]),
                    [np.cos(np.pi / 2), 0.0, 0.0, np.sin(np.pi / 2)],
                    atol=1e-12))

    # ---------------- PnP convention ----------------
    print('\nPNP CONVENTION (needs OpenCV)')
    try:
        import cv2
    except ImportError:
        print('  [SKIP] OpenCV not installed -- the marker modes are untested')
    else:
        s = 0.05
        obj = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]], float)
        obj_base = (Rb @ obj.T).T + Tb[:3, 3]
        K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])

        # The PHYSICAL camera: its pose as this repo stores it, then the same
        # pose in cv2's frame. Projecting through the cv one is what a D435i
        # does. Projecting through the repo's own frame is what an earlier
        # version of this test did -- and it passed, because it only ever
        # checked that pnp_pose inverts its own input.
        T_cam_mj = tools[1] @ X_true
        T_cam_cv = cam_mjocam_to_cv(T_cam_mj)
        R_cb = T_cam_cv[:3, :3].T
        t_cb = -R_cb @ T_cam_cv[:3, 3]
        pix, _ = cv2.projectPoints(obj_base, cv2.Rodrigues(R_cb)[0], t_cb,
                                   K, None)

        T_seed_mj = tools[1] @ transform_from([0.07, 0.01, 0.11],
                                              [0.05, -0.05, np.pi + 0.10])
        rec_cv, err = pnp_pose(obj_base, pix, K,
                               T_init=cam_mjocam_to_cv(T_seed_mj))
        # 1e-5 m, not 1e-6: SOLVEPNP_ITERATIVE stops on its own criteria and
        # lands ~1.5e-6 off on this synthetic case. Ten micrometres is two
        # orders finer than any calibration that matters here; the point of the
        # check is the frame, which is a whole rotation away from either answer.
        failures += not check(
            'solvePnP returns the camera pose in the OpenCV frame',
            np.allclose(rec_cv, T_cam_cv, atol=1e-5),
            'max diff %.2e, reprojection %.2e px'
            % (np.max(np.abs(rec_cv - T_cam_cv)), err))
        failures += not check(
            'converting it recovers the pose config stores',
            np.allclose(cam_cv_to_mjocam(rec_cv), T_cam_mj, atol=1e-5),
            'max diff %.2e'
            % np.max(np.abs(cam_cv_to_mjocam(rec_cv) - T_cam_mj)))

        # ---------------- eye-in-hand, end to end ----------------
        # One board, several arm poses, one constant mount: run_wrist's whole
        # arithmetic without a robot. If this holds, the only things left to go
        # wrong at the cell are the operator's poses and the clamp itself.
        print('\nWRIST RECOVERY FROM IMAGES')
        fitted = []
        for Tt in tools:
            T_c = cam_mjocam_to_cv(Tt @ X_true)
            R_c = T_c[:3, :3].T
            t_c = -R_c @ T_c[:3, 3]
            pix_i, _ = cv2.projectPoints(obj_base, cv2.Rodrigues(R_c)[0],
                                         t_c, K, None)
            rec_i, _e = pnp_pose(obj_base, pix_i, K,
                                 T_init=cam_mjocam_to_cv(T_seed_mj))
            fitted.append(wrist_mount(Tt, cam_cv_to_mjocam(rec_i)))
        X_fit, dt, dr = mean_transform(fitted)
        failures += not check(
            'recovers the mount from projected images of one fixed board',
            np.allclose(X_fit, X_true, atol=1e-6),
            'off %.2e m, spread %.1e mm / %.1e deg'
            % (np.linalg.norm(X_fit[:3, 3] - X_true[:3, 3]), dt * 1e3, dr))

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
