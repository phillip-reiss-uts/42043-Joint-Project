"""Camera extrinsic calibration: the two methods, as maths.

WHY THIS MODULE IS SEPARATE
---------------------------
Both methods produce the same thing -- the 4x4 pose of the camera in the robot
base frame -- but from different data. Keeping the solvers here (numpy, plus
OpenCV only for ArUco) means they can be tested offline, and the hardware I/O
lives in `tools/calibrate_camera.py`.

METHOD 1 — TOUCH  (`solve_touch`)
---------------------------------
The robot is the ruler. The operator places the cube, centres the closed
gripper over it and lowers until it just touches the top face. At that instant
the cube's top-face centre in the base frame is the gripper base's (x, y) --
the pad midpoint sits on the tool axis -- and its z is known by construction
(TABLE_HEIGHT + OBJECT_HEIGHT). Repeat at N >= 4 positions spread over the
table and fit the camera pose that makes the deprojected detections land on
those points.

This needs no OpenCV and no printed target. Its weakness is orientation: the
robot's hand is only as repeatable as the operator, so the fit relies on the
spread of the stations.

METHOD 2 — ARUCO  (`solve_aruco`, `board_pose_from_two_touches`)
---------------------------------------------------------------
Print a marker, lay it flat on the table, touch two adjacent corners with the
TCP. Those two points fix the marker's plane and yaw in the base frame, so
solvePnP against the detected corners yields the camera pose from a single
view. Better orientation than the touch method, but needs OpenCV and a board.

BOTH are only as good as the model they are fitted against: a wrong
TABLE_HEIGHT shifts every z, and a bent board cannot be rescued by maths.
"""

import numpy as np

from src.config import TABLE_HEIGHT


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


def transform_to_pos_target(T, table_z=TABLE_HEIGHT):
    # Convert a camera pose into the three numbers src/config.py stores.
    #
    # Returns (pos, target, up): the camera position, the point where its
    # optical axis meets the table plane, and the image-up direction. Feeding
    # these back through world_real.camera_transform() reproduces T exactly.
    T = np.asarray(T, float)
    pos = T[:3, 3].copy()
    forward = -T[:3, 2].copy()
    forward /= max(np.linalg.norm(forward), 1e-12)
    up = T[:3, 1].copy()
    if forward[2] < -1e-6:
        t = (table_z - pos[2]) / forward[2]
        if t > 0:
            return pos, pos + forward * t, up
    return pos, pos + forward * 1.0, up


# --------------------------------------------------------------------------
# METHOD 1: TOUCH
# --------------------------------------------------------------------------
class _FrameSource:
    # Minimal DepthSource over a stored frame, so the production detector can
    # run on a capture taken earlier without touching the camera again.

    def __init__(self, depth, K):
        self._depth = depth
        self._K = K

    def capture(self):
        return self._depth, self._K

    @property
    def size(self):
        return self._depth.shape[1], self._depth.shape[0]

    def close(self):
        pass


def detect_xy(T, depth, K):
    # Deproject + detect one frame with camera pose T; return the best detection
    # centre's (x, y) in the base frame, or None.
    #
    # This is exactly the production path (FixedCameraWorld), so the calibration
    # is fitted to the same detector the robot will use.
    from src.world_real import FixedCameraWorld
    world = FixedCameraWorld(_FrameSource(depth, K),
                             lambda: (np.zeros(3), np.eye(3)), T)
    views = world.perceive()
    if not views:
        return None
    return np.array(views[0].centre[:2], float)


def solve_touch(frames, targets, T0, detect=None, max_nfev=600):
    # Fit the camera pose so the detected cube centres land on the touched
    # points. Returns (T, rms_metres, n_used).
    #
    # frames  : list of (depth_uint16, K) captures
    # targets : list of (x, y) base-frame touch points, one per frame
    # T0      : initial guess (src/config.py's values)
    #
    # Only x and y are compared: the detector reports a fixed z
    # (TABLE_HEIGHT + half the cube), so z carries no information here.
    #
    # That also means the camera's HEIGHT is not observable from these
    # residuals: translating the camera along +z moves no base-frame x or y.
    # The height is therefore taken from T0 (measure it), and the remaining
    # five degrees of freedom -- x, y, and the orientation -- are solved here.
    # If the height is uncertain, use the ArUco method: PnP observes it.
    from scipy.optimize import least_squares

    if len(frames) != len(targets):
        raise ValueError('one target per frame')
    if len(frames) < 4:
        raise ValueError('need at least 4 stations (8 residuals for 5 params)')

    detect = detect or detect_xy
    targets = [np.asarray(t, float) for t in targets]
    z_fixed = float(np.asarray(T0, float)[2, 3])

    def residuals(p):
        T = transform_from([p[0], p[1], z_fixed], p[2:])
        out = []
        for (depth, K), tgt in zip(frames, targets):
            c = detect(T, depth, K)
            if c is None:
                out.extend([0.25, 0.25])      # a metre off: reject this station
            else:
                out.extend([c[0] - tgt[0], c[1] - tgt[1]])
        return np.asarray(out, float)

    p0 = np.r_[np.asarray(T0, float)[0, 3], np.asarray(T0, float)[1, 3],
               R_to_rotvec(np.asarray(T0, float)[:3, :3])]
    res = least_squares(residuals, p0, method='lm', max_nfev=max_nfev)
    T = transform_from([res.x[0], res.x[1], z_fixed], res.x[2:])
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    return T, rms, len(frames)


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
    obj = np.array([[0.0, 0.0, 0.0], [s, 0.0, 0.0],
                    [s, s, 0.0], [0.0, s, 0.0]], float)
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

    if T_init is not None:
        T_init = np.asarray(T_init, float)
        R_cb = T_init[:3, :3].T
        t_cb = -R_cb @ T_init[:3, 3]
        ok, rvec, tvec = cv2.solvePnP(
            obj, pix, K, dist, cv2.Rodrigues(R_cb)[0], t_cb.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, pix, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError('solvePnP failed')

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pix) ** 2,
                                       axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, float).reshape(3)
    # solvePnP returns X_cam = R X_base + t, so the camera pose in the base
    # frame is the inverse.
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T, err
