# Object perception from the wrist RGB-D camera.

from dataclasses import dataclass

import numpy as np

from src.config import (
    SELF_FILTER, SURFACE_BAND, DEPTH_MAX,
    TABLE_HEIGHT, TABLE_XY_BOUNDS,
    OBJECT_HEIGHT, OBJECT_SIZE,
)


@dataclass
class Detection:
    # One perceived object cluster.
    centre: np.ndarray      # (3,) world position of the object centre, or None
    n_points: int           # points in the accepted cluster (0 = not found)
    points: np.ndarray      # (N,3) world points of that cluster, or None
    yaw: float              # degrees modulo 90; 0.0 if undetectable
    span: np.ndarray = None # (2,) xy extent of the cluster, if computed
    score: float = 0.0      # quality score; higher = cleaner, denser cloud
    hull: np.ndarray = None # (M,3) top-face convex hull, the yaw's own data

    @property
    def ok(self):
        return self.centre is not None and self.n_points > 0


# GEOMETRY
def _hull2d(P):
    # Convex hull of (N,2) points, via Qhull (C speed). The old monotone chain
    # was pure Python and cost ~47 ms per call. scipy is imported lazily so the
    # observer's deferred-loading arrangement is preserved.
    from scipy.spatial import ConvexHull, QhullError
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 3:
        return P
    try:
        hull = ConvexHull(P)
    except QhullError:
        # degenerate (collinear) cloud: no rectangle fit downstream, same as
        # the old chain returning fewer than 3 points
        return np.zeros((0, 2))
    return hull.points[hull.vertices]


def _top_band(obj_pts, band):
    # Points within +-band of the cluster's median z, and their 2D hull.
    # Returns (band_points, hull_xy, z_of_band).
    if obj_pts is None or len(obj_pts) < 50:
        return None, None, None
    z_med = np.median(obj_pts[:, 2])
    top = obj_pts[np.abs(obj_pts[:, 2] - z_med) < band]
    if len(top) < 30:
        return top, None, z_med
    H = _hull2d(top[:, :2])
    if len(H) < 3:
        return top, None, z_med
    return top, H, float(np.median(top[:, 2]))


def _calipers(H):
    # Minimum-area rectangle of a 2D hull. Returns (centre_xy, yaw_deg,
    # half_extents) or None. The centre and the yaw come from the SAME edge,
    # which is the whole point of doing this once.
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        n = np.linalg.norm(e)
        if n < 1e-12:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = H @ u, H @ v
        w = pu.max() - pu.min()
        h = pv.max() - pv.min()
        area = w * h
        if best is None or area < best[0]:
            cu = 0.5 * (pu.max() + pu.min())
            cv = 0.5 * (pv.max() + pv.min())
            best = (area, u * cu + v * cv,
                    float(np.degrees(np.arctan2(u[1], u[0])) % 90.0),
                    np.array([w / 2.0, h / 2.0]))
    return None if best is None else best[1:]


# --- DISABLED: fitted_centre / detect_yaw. Superseded by _top_band + _calipers, which perceive_all now calls directly so the centre, the yaw and the drawn hull all come from one pass. Kept for reference. ---
# def fitted_centre(obj_pts, band=0.004):
#     # Centre of the minimum-area rectangle of the object's top face.
#     _, H, _ = _top_band(obj_pts, band)
#     if H is None:
#         return None
#     rect = _calipers(H)
#     return None if rect is None else rect[0]
#
#
# def detect_yaw(obj_pts, band=0.004):
#     # Orientation of a box-like object, degrees modulo 90.
#     if obj_pts is None or len(obj_pts) < 50:
#         return None, 0, 0
#     top, H, _ = _top_band(obj_pts, band)
#     if H is None:
#         return None, 0 if top is None else len(top), 0
#     rect = _calipers(H)
#     if rect is None:
#         return None, len(top), len(H)
#     return rect[1], len(top), len(H)
#
#
def rotation_with_yaw(yaw_deg, R_down):
    # R_down rotated about world z, so the approach stays down and the jaw turns.
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz @ R_down


# SENSING
# The valid-object region is TABLE_XY_BOUNDS from config, so the simulation and the
# hardware share one definition; the mask itself lives in camera.py.


def _filter_points(model, data, camera, cam_id):
    # Shared point filter: world points above the surface, in range, finite, and
    # INSIDE THE TABLE FOOTPRINT. The rule itself is in camera.py so there is one
    # copy of it -- height alone used to let the robot base and column through
    # whenever they came into view.
    depth = camera.capture_depth(data)
    return camera.surface_points(depth, data, SELF_FILTER, DEPTH_MAX,
                                 TABLE_HEIGHT + SURFACE_BAND,
                                 xy_bounds=TABLE_XY_BOUNDS)


def _span_ok(span):
    # True if the cluster's xy extent matches a single object.
    size = float(np.asarray(OBJECT_SIZE)[0]) * 2.0
    return (not np.any(span > size * np.sqrt(2.0) * 1.10)
            and not np.any(span < size * 0.70))


def score_cluster(span, n_points):
    # Quality score: span fidelity x log point count.
    size = float(np.asarray(OBJECT_SIZE)[0]) * 2.0
    fidelity = min(1.0, min(span) / size)
    return fidelity * np.log(max(n_points, 1.0))


def cluster_objects(points, voxel=0.02):
    # Split a point cloud into connected components. Voxelise, then label with
    # scipy.ndimage in C. 6-connectivity matches the old face-neighbour flood
    # fill exactly, so the components are unchanged.
    if points is None or len(points) == 0:
        return []

    from scipy import ndimage

    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    keys = keys - keys.min(axis=0)
    grid = np.zeros(keys.max(axis=0) + 1, dtype=np.uint8)
    grid[keys[:, 0], keys[:, 1], keys[:, 2]] = 1

    struct = np.zeros((3, 3, 3), dtype=np.uint8)
    struct[1, 1, 1] = 1
    struct[0, 1, 1] = struct[2, 1, 1] = 1
    struct[1, 0, 1] = struct[1, 2, 1] = 1
    struct[1, 1, 0] = struct[1, 1, 2] = 1

    labels, n = ndimage.label(grid, structure=struct)
    ids = labels[keys[:, 0], keys[:, 1], keys[:, 2]]
    clusters = [points[ids == c] for c in range(1, n + 1)]
    clusters.sort(key=len, reverse=True)
    return clusters


def perceive_all(model, data, camera, cam_id):
    # Detect every object on the surface.
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

        # One band-and-hull pass feeds the centre, the yaw and the stored hull,
        # so the drawn hull is the one the yaw was actually fitted to.
        _, H, z_band = _top_band(pts, 0.004)
        rect = _calipers(H) if H is not None else None

        if rect is not None:
            centre_xy = rect[0]
            yaw = rect[1]
        else:
            centre_xy = 0.5 * (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0))
            yaw = 0.0
        centre = np.array([centre_xy[0], centre_xy[1],
                           TABLE_HEIGHT + OBJECT_HEIGHT / 2.0])

        hull = None
        if H is not None:
            hull = np.column_stack([H, np.full(len(H), z_band)])

        dets.append(Detection(
            centre, len(pts), pts,
            yaw if yaw is not None else 0.0,
            span=span,
            score=score_cluster(span, len(pts)),
            hull=hull,
        ))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def perceive(model, data, camera, cam_id):
    # Single-object path: best cluster from perceive_all, or empty Detection.
    dets = perceive_all(model, data, camera, cam_id)
    if not dets:
        return Detection(None, 0, None, 0.0)
    return dets[0]
