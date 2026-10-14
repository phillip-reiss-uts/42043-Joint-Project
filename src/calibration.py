# Camera intrinsic calibration from rendered views of a planar marker board.

import math

import mujoco
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from src.config import (
    # LEFTOVER: CALIB_BOARD_THICK was imported but never read.
    CALIB_MARKERS, CALIB_SPACING, CALIB_MARKER_R,
)


# BOARD
def board_marker_local():
    # Board-frame (X, Y) of every marker centre, in row-major grid order.
    c = 0.5 * (CALIB_MARKERS - 1)
    pts = []
    for i in range(CALIB_MARKERS):
        for k in range(CALIB_MARKERS):
            pts.append([(i - c) * CALIB_SPACING, (k - c) * CALIB_SPACING])
    return np.array(pts, float)


# --- DISABLED: board_marker_world, never referenced. ---
# def board_marker_world(body_pos, body_mat, z_local=None):
#     # World positions of the markers for a given board pose.
#     if z_local is None:
#         z_local = 0.5 * CALIB_BOARD_THICK
#     loc = board_marker_local()
#     n = loc.shape[0]
#     pts = np.column_stack([loc, np.full(n, z_local)])
#     return pts @ body_mat.T + body_pos
#
#
# # DETECTION
def detect_markers(rgb, depth, z_hint, min_area=4, max_area=400,
                   thresh=200, depth_tol=0.05):
    # Find marker blob centroids in a rendered frame.
    bright = (rgb[:, :, 0] > thresh) & (rgb[:, :, 1] > thresh) \
        & (rgb[:, :, 2] > thresh)
    # reject background: a marker sits on the board plane
    if depth is not None:
        near = np.isfinite(depth) & (np.abs(depth - z_hint) < depth_tol)
        bright = bright & near

    labels, n = ndimage.label(bright)
    if n == 0:
        return np.zeros((0, 2))

    idx = np.arange(1, n + 1)
    areas = ndimage.sum(bright, labels, idx)
    keep = (areas >= min_area) & (areas <= max_area)

    w = rgb[:, :, 0].astype(float) * bright
    coms = np.array(ndimage.center_of_mass(w, labels, idx))
    # center_of_mass returns (row, col) = (v, u)
    out = coms[keep][:, ::-1]
    return out


# HOMOGRAPHY
def homography_dlt(src, dst):
    # H mapping src (N,2) -> dst (N,2), least squares via the DLT.
    def normalise(p):
        m = p.mean(axis=0)
        s = np.sqrt(2.0) / (np.linalg.norm(p - m, axis=1).mean() + 1e-12)
        T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])
        q = (p - m) * s
        return q, T

    s_n, T_s = normalise(np.asarray(src, float))
    d_n, T_d = normalise(np.asarray(dst, float))

    n = s_n.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = s_n[i]
        u, v = d_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_d) @ Hn @ T_s
    return H / H[2, 2]


def apply_h(H, pts):
    # Project (N,2) through H.
    p = np.column_stack([pts, np.ones(len(pts))])
    q = p @ H.T
    return q[:, :2] / q[:, 2:3]


def _order_ccw(p):
    # Cyclic ordering of a point set about its centroid.
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)]


def _extreme_quad(blobs):
    # The four extreme blobs: corners of a convex grid under a projective map.
    s = blobs[:, 0] + blobs[:, 1]
    d = blobs[:, 0] - blobs[:, 1]
    ext = np.array([blobs[np.argmin(s)], blobs[np.argmax(s)],
                    blobs[np.argmin(d)], blobs[np.argmax(d)]])
    if len(np.unique(np.round(ext, 3), axis=0)) < 4:
        return None
    return ext


def _label_with(H, blobs, board_xy, tol):
    # Predict every board marker through H (board -> image) and match blobs.
    pred = apply_h(H, board_xy)
    bp, mp, used = [], [], set()
    for j, pj in enumerate(pred):
        dist = np.linalg.norm(blobs - pj, axis=1)
        m = int(np.argmin(dist))
        if dist[m] < tol and m not in used:
            used.add(m)
            bp.append(board_xy[j])
            mp.append(blobs[m])
    if not bp:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.array(bp), np.array(mp)


def label_blobs(blobs, board_xy, tol=8.0):
    # Assign every detected blob a board marker index, without using K.
    n = CALIB_MARKERS
    if len(blobs) < 8:
        return None, None

    # board corners in CCW order seen from +z (x right, y up): BL, BR, TR, TL
    corners = np.array([
        board_xy[0],                # (0, 0)
        board_xy[(n - 1) * n],      # (n-1, 0)
        board_xy[-1],               # (n-1, n-1)
        board_xy[n - 1],            # (0, n-1)
    ], float)

    ext = _extreme_quad(blobs)
    if ext is None:
        return None, None

    ordered = _order_ccw(ext)
    best = None
    for img_pts in (ordered, ordered[::-1]):
        for shift in range(4):
            ip = np.roll(img_pts, shift, axis=0)
            try:
                # board -> image, which is the direction needed to predict
                H = homography_dlt(corners, ip)
            except np.linalg.LinAlgError:
                continue
            bp, mp = _label_with(H, blobs, board_xy, tol)
            if best is None or len(bp) > len(best[0]):
                best = (bp, mp)

    if best is None or len(best[0]) < 6:
        return None, None

    bp, mp = best
    for _ in range(2):
        H = homography_dlt(bp, mp)
        bp, mp = _label_with(H, blobs, board_xy, tol)
        if len(bp) < 6:
            return None, None
    return bp, mp


# ZHANG
def _v_ij(H, i, j):
    # Zhang's v_ij vector built from the columns of H.
    h = H
    return np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j],
    ])


def intrinsics_from_homographies(Hs, verbose=False):
    # Solve B from the Zhang constraints, then extract K. None if degenerate.
    V = []
    for H in Hs:
        V.append(_v_ij(H, 0, 1))
        V.append(_v_ij(H, 0, 0) - _v_ij(H, 1, 1))
    V = np.array(V)
    if V.shape[0] < 6:
        return None
    _, sv, Vt = np.linalg.svd(V)

    # B is defined only up to scale; fix the gauge so that B11 > 0.
    b = Vt[-1]
    if b[0] < 0:
        b = -b
    B11, B12, B22, B13, B23, B33 = b

    if verbose:
        print('   constraint singular values: %s'
              % np.round(sv, 4))
        print('   b = %s' % np.round(b, 6))

    denom = B11 * B22 - B12 ** 2
    if abs(denom) < 1e-14 or abs(B11) < 1e-14:
        return None

    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if lam / B11 <= 0 or lam * B11 / denom <= 0:
        return None

    fx = math.sqrt(lam / B11)
    fy = math.sqrt(lam * B11 / denom)
    skew = -B12 * fx ** 2 * fy / lam
    u0 = skew * v0 / fy - B13 * fx ** 2 / lam

    return np.array([[fx, skew, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])


# REFINEMENT
def _project(K, board_pts, rvec, tvec):
    # Project board points with Rodrigues rotation, into the Z-forward frame.
    R, _ = _rodrigues(rvec)
    bp = np.asarray(board_pts, float)
    if bp.shape[1] == 2:
        bp = np.column_stack([bp, np.zeros(len(bp))])
    Xc = bp @ R.T + tvec
    u = K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]
    return np.column_stack([u, v])


def _rodrigues(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3), np.zeros(3)
    k = rvec / theta
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(theta) * Kx + (1 - math.cos(theta)) * (Kx @ Kx)
    return R, k


def refine(K0, views):
    # Bundle-adjust intrinsics and per-view extrinsics on reprojection error.
    nv = len(views)
    # initial extrinsics from a pose guess using K0
    p0 = [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]
    for bp, ip in views:
        rvec, tvec = _initial_pose(K0, bp, ip)
        p0 += list(rvec) + list(tvec)

    def unpack(p):
        K = np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
        return K, [np.array(p[4 + 6 * i: 10 + 6 * i]) for i in range(nv)]

    def resid(p):
        K, poses = unpack(p)
        out = []
        for i, (bp, ip) in enumerate(views):
            rvec, tvec = poses[i][:3], poses[i][3:]
            proj = _project(K, bp, rvec, tvec)
            out.append((proj - ip).ravel())
        return np.concatenate(out)

    lo = [10.0, 10.0, 0.0, 0.0] + [-np.inf] * (6 * nv)
    hi = [1e5, 1e5, 1e4, 1e4] + [np.inf] * (6 * nv)
    sol = least_squares(resid, p0, bounds=(lo, hi), method='trf',
                        x_scale='jac', max_nfev=4000)
    K, poses = unpack(sol.x)
    return K, sol, resid(sol.x)


def _initial_pose(K, board_pts, image_pts):
    # Extrinsics from the homography between the board plane and the image.
    H = homography_dlt(board_pts, image_pts)
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = Kinv @ H[:, 0], Kinv @ H[:, 1], Kinv @ H[:, 2]
    lam = 2.0 / (np.linalg.norm(h1) + np.linalg.norm(h2))
    r1, r2, t = lam * h1, lam * h2, lam * h3
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    # rvec from R
    theta = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if theta < 1e-9:
        return np.zeros(3), t
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2 * math.sin(theta))
    return axis * theta, t


# RIG
def capture_views(model, data, n_views=12, width=640, height=480,
                  distances=(0.36, 0.42), seed=0, rms_tol=0.5,
                  tilt_range=(12.0, 30.0)):
    # Teleport the board to varied poses and collect marker detections.
    from src.camera import RGBDCamera

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'calib_board')
    if bid < 0:
        raise SystemExit('model has no calib_board; build with '
                         'load_robot_model(with_calibration_board=True)')

    n_markers = CALIB_MARKERS * CALIB_MARKERS
    cam = RGBDCamera(model, 'depth_cam', width=width, height=height)
    rng = np.random.default_rng(seed)
    board_xy = board_marker_local()

    # camera axes in world (MuJoCo: view direction is local -z)
    Rc = data.cam_xmat[cam.cam_id].reshape(3, 3)
    ax_x, ax_y, ax_z = Rc[:, 0], Rc[:, 1], Rc[:, 2]
    view_dir = -ax_z

    views = []
    arm0 = np.array(data.qpos[:6])
    for v in range(n_views):
        dist = distances[v % len(distances)]
        # Tilt separates focal length from distance (see the module notes).
        tilt = np.deg2rad(rng.uniform(*tilt_range))
        roll = np.deg2rad(rng.uniform(-20.0, 20.0))
        lat_x = rng.uniform(-0.04, 0.04)
        lat_y = rng.uniform(-0.03, 0.03)

        # board +z faces the camera: its normal is -view_dir
        n = -view_dir
        # build an orthonormal board frame then rotate by the tilt/roll
        t1 = np.cross(n, ax_x)
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(n, ax_y)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        def rot(axis, ang):
            Kx = np.array([[0, -axis[2], axis[1]],
                           [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
            return np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * (Kx @ Kx)

        Rb = rot(t1, tilt) @ rot(n, roll) @ np.column_stack([t1, t2, n])

        pos = (data.cam_xpos[cam.cam_id] + dist * view_dir
               + lat_x * ax_x + lat_y * ax_y)

        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rb.ravel())

        adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              'calib_board_free')]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = q
        data.qpos[:6] = arm0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        rgb, depth = cam.capture(data)
        blobs = detect_markers(rgb, depth, z_hint=dist)
        bp, ip = label_blobs(blobs, board_xy)

        rms = None
        if bp is not None and len(bp) == n_markers:
            H = homography_dlt(bp, ip)
            rms = float(np.sqrt(np.mean(
                np.sum((apply_h(H, bp) - ip) ** 2, axis=1))))
            if rms > rms_tol:
                bp = None

        if bp is not None and len(bp) == n_markers:
            views.append((bp, ip))
            print('  view %2d: dist %.2f m  tilt %4.1f deg  blobs %2d  '
                  'matched %2d/%d  H rms %.3f px  ACCEPTED'
                  % (v, dist, math.degrees(tilt), len(blobs), len(bp),
                     n_markers, rms))
        else:
            print('  view %2d: dist %.2f m  tilt %4.1f deg  blobs %2d  '
                  'matched %s/%d  %s  REJECTED'
                  % (v, dist, math.degrees(tilt), len(blobs),
                     len(bp) if bp is not None else 0, n_markers,
                     ('H rms %.3f px' % rms) if rms is not None else 'incomplete'))

    cam.renderer.close()
    return views


def main():
    from src.model_loader import load_robot_model

    model = load_robot_model(with_calibration_board=True)
    if model is None:
        raise SystemExit('model failed to load')

    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    print()
    print('=' * 74)
    print('CAMERA INTRINSIC CALIBRATION (Zhang, from rendered images)')
    print('=' * 74)
    print('board        : %dx%d markers, %.0f mm pitch, %.1f mm radius'
          % (CALIB_MARKERS, CALIB_MARKERS, CALIB_SPACING * 1000,
             CALIB_MARKER_R * 1000))
    print('target poses : varied distance, tilt and lateral offset')
    print()

    views = capture_views(model, data, n_views=16)
    print()
    print('accepted views: %d' % len(views))
    if len(views) < 3:
        raise SystemExit('need at least 3 views')

    Hs = [homography_dlt(bp, ip) for bp, ip in views]
    K_lin = intrinsics_from_homographies(Hs, verbose=True)
    if K_lin is None:
        raise SystemExit('linear solve failed')
    print()
    print('closed-form from B (Zhang), before refinement:')
    print('   fx %.2f  fy %.2f  cx %.2f  cy %.2f'
          % (K_lin[0, 0], K_lin[1, 1], K_lin[0, 2], K_lin[1, 2]))

    K, sol, resid = refine(K_lin, views)
    rms = math.sqrt(np.mean(resid ** 2))
    print()
    print('after nonlinear refinement:')
    print('   fx %.3f  fy %.3f  cx %.3f  cy %.3f'
          % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print('   reprojection RMS: %.4f px   (max %.4f px)'
          % (rms, np.abs(resid).max()))

    # ground truth, read only now, purely for comparison
    fovy = math.radians(model.cam_fovy[0])
    f_true = 0.5 * 480 / math.tan(fovy / 2.0)
    print()
    print('=' * 74)
    print('vs the simulator ground truth (cam_fovy = %.1f deg, 640x480):'
          % model.cam_fovy[0])
    print('   %-8s %-12s %-12s %s' % ('', 'calibrated', 'truth', 'error'))
    for name, got, tru in (('fx', K[0, 0], f_true), ('fy', K[1, 1], f_true),
                           ('cx', K[0, 2], 320.0), ('cy', K[1, 2], 240.0)):
        print('   %-8s %-12.3f %-12.3f %+.4f  (%.3f%%)'
              % (name, got, tru, got - tru, 100 * (got - tru) / tru))
    print()
    print('   camera.py represents this as K = [[-fx, 0, cx], [0, fy, cy]];')
    print('   the sign is MuJoCo\'s look-down--z convention, not a result.')


if __name__ == '__main__':
    main()
