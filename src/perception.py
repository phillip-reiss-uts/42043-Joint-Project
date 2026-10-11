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
    """One perceived object cluster.

    Replaces the (estimate, n_pts, obj_pts) triple, which call sites unpacked as
    `est, n, _` -- silently discarding the point array that detect_yaw needs.
    """
    centre: np.ndarray      # (3,) world position of the object centre, or None
    n_points: int           # points in the accepted cluster (0 = not found)
    points: np.ndarray      # (N,3) world points of that cluster, or None
    yaw: float              # degrees modulo 90; 0.0 if undetectable
    span: np.ndarray = None # (2,) xy extent of the cluster, if computed
    score: float = 0.0      # quality score; higher = cleaner, denser cloud

    @property
    def ok(self):
        return self.centre is not None and self.n_points > 0


# --------------------------------------------------------------- geometry ---
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


def fitted_centre(obj_pts, band=0.004):
    """Centre of the minimum-area rectangle of the object's top face.

    Preferred over an axis-aligned bbox centre, which leans toward whichever
    side is more complete when the cluster is clipped or the object is rotated.
    Uses the same rotating-calipers fit as detect_yaw.
    """
    z_med = np.median(obj_pts[:, 2])
    top = obj_pts[np.abs(obj_pts[:, 2] - z_med) < band]
    if len(top) < 30:
        return None

    H = _hull2d(top[:, :2])
    if len(H) < 3:
        return None

    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        n = np.linalg.norm(e)
        if n < 1e-12:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = H @ u, H @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0]:
            cu = 0.5 * (pu.max() + pu.min())
            cv = 0.5 * (pv.max() + pv.min())
            best = (area, u * cu + v * cv)

    return best[1] if best else None


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


# ---------------------------------------------------------------- sensing ---
def _filter_points(model, data, camera, cam_id):
    """Shared point filter: world points above the surface, in range, finite."""
    rgb, depth = camera.capture(data)
    points = camera.deproject(depth, data)

    cam_pos = data.cam_xpos[cam_id]
    near = np.linalg.norm(points - cam_pos, axis=-1)
    keep = (np.isfinite(points).all(axis=-1)
            & (near > SELF_FILTER)
            & (near < DEPTH_MAX)
            & (points[..., 2] > TABLE_HEIGHT + SURFACE_BAND))
    return points[keep]


def _span_ok(span):
    """True if the cluster's xy extent matches a single object.

    A clean top face spans 2*half-extent (0.060) rising to the diagonal
    (0.060*sqrt2 = 0.085) at 45 deg yaw. Larger means merged objects or table;
    smaller means a clipped view. Both bias the centre, so both are rejected.
    """
    size = float(np.asarray(OBJECT_SIZE)[0]) * 2.0
    return (not np.any(span > size * np.sqrt(2.0) * 1.10)
            and not np.any(span < size * 0.70))


def score_cluster(span, n_points):
    """Quality score: span fidelity x log point count.

    A clean object has span ~0.060-0.085 and 3000-5000 points; a clipped one has
    a short span and fewer points. Fidelity dominates, so a partially visible
    object never outranks a fully visible one just because it is nearer.
    """
    size = float(np.asarray(OBJECT_SIZE)[0]) * 2.0
    fidelity = min(1.0, min(span) / size)
    return fidelity * np.log(max(n_points, 1.0))


def cluster_objects(points, voxel=0.02):
    """Split a point cloud into connected components.

    Voxel-grid flood fill, 6-connectivity. Correct when objects do not touch:
    MIN_OBJECT_SEPARATION guarantees that, which is why the span guard is a
    reliable per-cluster check rather than a heuristic.

    Returns a list of (N,3) point arrays, largest first.
    """
    if points is None or len(points) == 0:
        return []

    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    voxels = {}
    for i, k in enumerate(keys):
        voxels.setdefault(tuple(k), []).append(i)

    clusters = []
    seen = set()
    for v in voxels:
        if v in seen:
            continue
        # flood fill
        stack = [v]
        seen.add(v)
        idx = []
        while stack:
            c = stack.pop()
            idx.extend(voxels[c])
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                      (0, 0, 1), (0, 0, -1)):
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb in voxels and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        clusters.append(points[idx])

    clusters.sort(key=len, reverse=True)
    return clusters


def perceive_all(model, data, camera, cam_id):
    """Detect every object on the surface.

    The single-object perceive() assumed exactly one cluster; with a pool this
    becomes cluster + span-guard + score per connected component. Selection
    among objects is then `max(dets, key=lambda d: d.score)`.

    Returns a list of Detection, possibly empty.
    """
    obj = _filter_points(model, data, camera, cam_id)
    if len(obj) < 50:
        return []

    dets = []
    for pts in cluster_objects(obj):
        if len(pts) < 50:
            continue
        span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        if not _span_ok(span):
            continue

        centre_xy = fitted_centre(pts)
        if centre_xy is None:
            centre_xy = 0.5 * (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0))
        centre = np.array([centre_xy[0], centre_xy[1],
                           TABLE_HEIGHT + OBJECT_HEIGHT / 2.0])

        yaw, _, _ = detect_yaw(pts)
        dets.append(Detection(
            centre, len(pts), pts,
            yaw if yaw is not None else 0.0,
            span=span,
            score=score_cluster(span, len(pts)),
        ))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def perceive(model, data, camera, cam_id):
    """Single-object path: best cluster from perceive_all, or empty Detection.

    Kept so the original single-object pick_and_place flow keeps working while
    the multi-object flow is built out.
    """
    dets = perceive_all(model, data, camera, cam_id)
    if not dets:
        return Detection(None, 0, None, 0.0)
    return dets[0]