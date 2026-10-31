"""Camera extrinsic calibration: the ArUco method, as maths.

WHY THIS MODULE IS SEPARATE
---------------------------
The solver produces the 4x4 pose of the camera in the robot base frame. Keeping
it here (numpy, plus OpenCV only for ArUco) means it can be tested offline, and
the hardware I/O lives in `tools/calibrate_camera.py`.

ARUCO  (`solve_aruco`, `board_pose_from_two_touches`)
-----------------------------------------------------
Print a marker, lay it flat on the table, touch two adjacent corners with the
TCP. Those two points fix the marker's plane and yaw in the base frame, so
solvePnP against the detected corners yields the camera pose from a single
view. Every pose then gives one estimate of the same tool -> camera constant.

The fit is only as good as the model it is fitted against: a wrong TABLE_HEIGHT
shifts every z, and a bent board cannot be rescued by maths.
"""

import numpy as np


# --------------------------------------------------------------------------
# SMALL ROTATION HELPERS (numpy only; no cv2.Rodrigues dependency)
# --------------------------------------------------------------------------
def rotvec_to_R(v):
    # Rodrigues: rotation vector -> rotation matrix.
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def R_to_rotvec(R):
    # Inverse of the above, stable at 0 and pi.
    R = np.asarray(R, float)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        # Near pi: axis from the symmetric part.
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        if k[i] > 0:
            k = A[:, i] / k[i]
        return th * k / max(np.linalg.norm(k), 1e-12)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * w / (2.0 * np.sin(th))


def transform_from(pos, rotvec):
    # 4x4 from a translation and a rotation vector.
    T = np.eye(4)
    T[:3, :3] = rotvec_to_R(rotvec)
    T[:3, 3] = np.asarray(pos, float)
    return T


# --------------------------------------------------------------------------
# CAMERA FRAME CONVENTIONS
# --------------------------------------------------------------------------
# TWO conventions meet here and they differ by a half turn about the optical
# axis:
#
#     OpenCV / solvePnP    x right, y DOWN,  z FORWARD, into the scene
#     MuJoCo / this repo   x right, y UP,    z BACKWARD, out of the scene
#
# Everything downstream of a calibration is in the MuJoCo one:
# world_points_from_depth consumes it, RealObjectWorld composes it from FK(q) @
# T_flange_cam, and config's CAM_MOUNT_QUAT stores it. camera.py encodes the same
# choice in its intrinsics -- it builds K with a NEGATIVE fx for exactly this
# reason. cv2 is the opposite.
#
# Mixing them up yields a camera that looks backwards. That fails loudly, but
# only at the cell, and only after the numbers have been pasted into config.
#
# The conversion is a rotation of 180 deg about the optical axis, applied on the
# RIGHT, because it re-expresses the camera's OWN frame rather than the base
# frame: T_mj = T_cv @ C. It is its own inverse.
CAM_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def cam_cv_to_mjocam(T):
    # solvePnP/OpenCV camera pose -> the convention this repo stores.
    T = np.asarray(T, float).copy()
    T[:3, :3] = T[:3, :3] @ CAM_CV_TO_MJ
    return T


def cam_mjocam_to_cv(T):
    # The same transform, named for the other direction. It is an involution, so
    # this is not a separate operation -- only a separate call site.
    return cam_cv_to_mjocam(T)


def quat_from_rot(R):
    # Rotation matrix -> (w, x, y, z), the order arm_dynamics.rot_from_quat
    # takes, so a printed value can be pasted straight into config.
    R = np.asarray(R, float)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], float)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q if q[0] >= 0.0 else -q          # canonical sign: w >= 0


def wrist_mount(T_base_tool, T_base_cam):
    # Eye-in-hand: the constant transform from the controlled tool frame to the
    # camera. One arm pose is enough to state it; several are averaged.
    #
    # Both arguments must be in the SAME base frame, which they are when the
    # tool pose comes from the same FK that located the board. A rotation of the
    # base frame -- config's UR_ROOT_YAW, for one -- cancels in the product
    # inv(R T_tool) (R T_cam) = inv(T_tool) T_cam, so it does not matter which
    # frame the calibration was run in.
    return np.linalg.inv(np.asarray(T_base_tool, float)) \
        @ np.asarray(T_base_cam, float)


def mean_transform(Ts):
    # Chordal mean of a set of 4x4 poses: mean translation, and the rotation
    # closest (Frobenius) to the mean of the rotation matrices.
    #
    # Returns (T_mean, worst translation deviation in metres, worst rotation
    # deviation in degrees). The deviations are the honest error bar: they say
    # how well one constant describes every pose, which is what "the mount does
    # not move" means. Averaging quaternions instead would need a hemisphere
    # flip per term and is no better here.
    if not Ts:
        raise ValueError('mean_transform needs at least one pose')
    Ts = [np.asarray(T, float) for T in Ts]
    t = np.mean(np.array([T[:3, 3] for T in Ts]), axis=0)

    M = np.zeros((3, 3))
    for T in Ts:
        M += T[:3, :3]
    U, _s, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))   # keep det +1, never a reflection
    R = U @ D @ Vt

    T_mean = np.eye(4)
    T_mean[:3, :3] = R
    T_mean[:3, 3] = t

    worst_t = max(float(np.linalg.norm(T[:3, 3] - t)) for T in Ts)
    worst_r = 0.0
    for T in Ts:
        c = float(np.clip((np.trace(R.T @ T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        worst_r = max(worst_r, float(np.degrees(np.arccos(c))))
    return T_mean, worst_t, worst_r


# --------------------------------------------------------------------------
# METHOD 2: ARUCO
# --------------------------------------------------------------------------
def board_pose_from_two_touches(a, b, up=(0.0, 0.0, 1.0)):
    # Marker frame from two adjacent corners touched with the TCP.
    #
    # `a` is the origin corner, `b` the next one along the marker's +x edge.
    # The marker lies flat, so +z is world up and +y follows.
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    x = b - a
    x[2] = 0.0
    n = np.linalg.norm(x)
    if n < 1e-6:
        raise ValueError('the two touched corners are coincident')
    x /= n
    z = np.asarray(up, float)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)

    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = a
    return T


def aruco_corners(image_bgr, dict_name='DICT_4X4_50'):
    # Detect markers; returns [(id, (4,2) pixel corners)].
    #
    # OpenCV 4.7 moved detection behind ArucoDetector; older builds expose
    # detectMarkers directly. Support both.
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(image_bgr)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, d)
    if ids is None:
        return []
    return [(int(i), np.asarray(c, float).reshape(4, 2))
            for i, c in zip(np.ravel(ids), corners)]


def solve_aruco(image_bgr, K, T_board, marker_size, dict_name='DICT_4X4_50',
                dist=None, T_init=None):
    # Camera pose from one image of a marker whose pose in the base frame is
    # known (see board_pose_from_two_touches).
    #
    # Returns (T_base_cam, reprojection_rms_px).
    import cv2

    found = aruco_corners(image_bgr, dict_name)
    if not found:
        raise RuntimeError('no marker detected -- check lighting and focus')

    s = float(marker_size)
    # THE SQUARE IS ON THE -y SIDE. cv2 returns the marker's corners CLOCKWISE
    # as seen from the camera. The board frame has +z up and +x along the
    # touched corner 0 -> corner 1 edge, so clockwise is the -y side of that
    # edge: z cross x points OUT of the marker, not into it. Putting the
    # square on +y (as this did) mirrors the correspondence. solvePnP then
    # reprojects the image just as well but places the camera UNDER the board --
    # which is why every wrist run returned a camera hundreds of mm (once
    # 100 km) from the tool with a 100+ deg spread, and why pnp_pose's
    # above-the-board filter now refuses those poses outright.
    obj = np.array([[0.0, 0.0, 0.0], [s, 0.0, 0.0],
                    [s, -s, 0.0], [0.0, -s, 0.0]], float)
    obj_base = (np.asarray(T_board, float)[:3, :3] @ obj.T).T \
        + np.asarray(T_board, float)[:3, 3]

    # The corner order cv2 returns encodes the marker's own orientation -- the
    # printed bits break the square's 90-degree symmetry. Do NOT re-order the
    # corners to "try all four": every rotation of a square correspondence fits
    # the image equally well, and the search would silently return a pose
    # rotated 90 degrees about the marker normal. Instead the operator is shown
    # which detected corner is 0 (see tools/calibrate_camera.py).
    best = None
    for _mid, pix in found:
        T, err = pnp_pose(obj_base, pix, K, dist=dist, T_init=T_init)
        if best is None or err < best[0]:
            best = (err, T)

    if best is None:
        raise RuntimeError('solvePnP failed')
    return best[1], best[0]


def pnp_pose(obj_base, pix, K, dist=None, T_init=None):
    # Camera pose in the base frame from 3D base points and their pixels.
    #
    # `T_init` matters. A planar target has a mirror solution that reprojects
    # identically, and IPPE will happily return either. Seeding with the
    # approximate camera pose (from config) and iterating picks the physically
    # correct one -- the camera above the table looking down.
    import cv2

    obj = np.asarray(obj_base, float)
    pix = np.asarray(pix, float).reshape(-1, 2)
    K = np.asarray(K, float)

    # Candidate poses from BOTH solvers. The seeded iterative one is good when
    # the config is close, but it can wander when the board guess is off; IPPE
    # returns both planar solutions, one of which is the mirror. Every candidate
    # is then checked PHYSICALLY, not just by reprojection error -- that is what
    # stops a mirror or a wandered solution reaching the report as a 100 km
    # camera.
    cands = []
    if T_init is not None:
        T_init = np.asarray(T_init, float)
        R_cb = T_init[:3, :3].T
        t_cb = -R_cb @ T_init[:3, 3]
        ok, rvec, tvec = cv2.solvePnP(
            obj, pix, K, dist, cv2.Rodrigues(R_cb)[0], t_cb.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            cands.append((rvec, tvec))
    ok = False
    if hasattr(cv2, 'solvePnPGeneric'):
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj, pix, K, dist, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            ok = False
    if ok:
        cands.extend(zip(rvecs, tvecs))
    elif not cands:
        ok, rvec, tvec = cv2.solvePnP(obj, pix, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE)
        if ok:
            cands.append((rvec, tvec))
    if not cands:
        raise RuntimeError('solvePnP failed')

    z_board = float(np.mean(obj[:, 2]))
    best = None
    for rvec, tvec in cands:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - pix) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, float).reshape(3)
        # The mirror solution reprojects identically, so it is rejected here:
        # a camera photographing a marker on the table must be ABOVE the board
        # and must have every marker point in front of it.
        in_front = bool(np.all((obj @ R.T + t)[:, 2] > 0.0))
        p_cam = -R.T @ t
        if not (in_front and p_cam[2] > z_board + 0.02):
            continue
        if best is None or err < best[0]:
            T = np.eye(4)
            T[:3, :3] = R.T
            T[:3, 3] = p_cam
            best = (err, T)

    if best is None:
        raise RuntimeError('solvePnP found no physically valid pose '
                           '(camera below the board or behind it)')
    return best[1], best[0]
