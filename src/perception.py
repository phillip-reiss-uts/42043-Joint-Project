"""Object perception from the wrist RGB-D camera.

Pure functions over (model, data, camera, cam_id) -- nothing here moves the arm
or touches control inputs.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from src.config import (
    SELF_FILTER, SURFACE_BAND, DEPTH_MAX,
    TABLE_HEIGHT, OBJECT_HEIGHT, OBJECT_SIZE,
)


@dataclass
class Detection:
    """One perception result.

    Replaces the (estimate, n_pts, obj_pts) triple, which call sites unpacked as
    `est, n, _` -- silently discarding the point array that detect_yaw needs.
    """
    centre: np.ndarray      # (3,) world position of the object centre, or None
    n_points: int           # points in the accepted cluster (0 = not found)
    points: np.ndarray      # (N,3) world points of that cluster, or None
    yaw: float              # degrees modulo 90; 0.0 if undetectable

    @property
    def ok(self):
        return self.centre is not None and self.n_points > 0


def _hull2d(P):
    """Convex hull of (N,2) points, monotone chain."""
    P = np.unique(np.round(P, 9), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(pts):
        st = []
        for p in pts:
            while len(st) >= 2 and cross(st[-2], st[-1], p) <= 1e-12:
                st.pop()
            st.append(p)
        return st

    lo = half(P)
    up = half(P[::-1])
    return np.array(lo[:-1] + up[:-1])


def detect_yaw(obj_pts, band=0.004):
    """Orientation of a box-like object, degrees modulo 90.

    Uses the minimum-area enclosing rectangle of the top face. PCA is not usable
    here: a square face has isotropic 2D covariance, so its principal axis is
    arbitrary (measured error up to 45 deg). The face boundary carries the
    orientation instead.

    Returns (angle_deg_mod90, n_top, n_hull); angle is None if the top face has
    too few points to be reliable.
    """
    if obj_pts is None or len(obj_pts) < 50:
        return None, 0, 0

    z_med = np.median(obj_pts[:, 2])
    top = obj_pts[np.abs(obj_pts[:, 2] - z_med) < band]
    if len(top) < 30:
        return None, len(top), 0

    H = _hull2d(top[:, :2])
    if len(H) < 3:
        return None, len(top), len(H)

    best_area, best_angle = None, None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        n = np.linalg.norm(e)
        if n < 1e-12:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = H @ u, H @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best_area is None or area < best_area:
            best_area = area
            best_angle = float(np.degrees(np.arctan2(u[1], u[0])) % 90.0)

    return best_angle, len(top), len(H)


def rotation_with_yaw(yaw_deg, R_down):
    """R_down rotated about world z, so the approach stays down and the jaw turns.

    R_down is passed in rather than imported, so this module does not depend on
    the arm's rotation convention.
    """
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz @ R_down


def find_camera(model):
    """Return (name, id) of the wrist camera, or (None, -1)."""
    for candidate in ('camera/depth_cam', 'depth_cam'):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
        if cid >= 0:
            return candidate, cid
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and 'depth_cam' in name:
            return name, i
    return None, -1


def perceive(model, data, camera, cam_id):
    """Locate the object on the work surface.

    The floor plane is not fitted by RANSAC here: with the object close to a
    downward camera it has more inliers than the surface, and the fit locks onto
    the object's top face. The work surface height is known, so use it.

    Known limitation: this is a measurement function, not a search. It assumes
    the object is in view. From a raised/back pose it returns a table-sized
    cluster, which the span guard rejects -- it cannot find an object outside
    the camera footprint.
    """
    rgb, depth = camera.capture(data)
    points = camera.deproject(depth, data)

    cam_pos = data.cam_xpos[cam_id]
    near = np.linalg.norm(points - cam_pos, axis=-1)
    keep = (np.isfinite(points).all(axis=-1)
            & (near > SELF_FILTER)
            & (near < DEPTH_MAX)
            & (points[..., 2] > TABLE_HEIGHT + SURFACE_BAND))

    obj = points[keep]
    if len(obj) < 50:
        return Detection(None, 0, None, 0.0)

    # A point-count gate alone cannot tell the object from the table: an
    # out-of-view peek returned 19412 points -- the whole tabletop -- which
    # passed MIN_OBJECT_POINTS. Reject anything with a far larger footprint.
    obj_half = np.asarray(OBJECT_SIZE)
    span = obj[:, :2].max(axis=0) - obj[:, :2].min(axis=0)
    if np.any(span > 3.0 * obj_half[:2]):
        return Detection(None, len(obj), None, 0.0)

    centre_xy = 0.5 * (obj[:, :2].min(axis=0) + obj[:, :2].max(axis=0))

    # The camera only ever sees the top face, so the centroid's z is the top of
    # the object, not its centre. Derive the centre from the known surface height.
    centre = np.array([centre_xy[0], centre_xy[1],
                       TABLE_HEIGHT + OBJECT_HEIGHT / 2.0])

    yaw, _, _ = detect_yaw(obj)

    return Detection(centre, len(obj), obj, yaw if yaw is not None else 0.0)