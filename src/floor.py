# --- DISABLED: this whole module is unreferenced. Nothing imports it;
# fit_plane_ransac / load_plane / save_plane / remove_plane / remove_self
# have no callers. Retained commented out rather than deleted. ---
# # src/floor.py
# # Floor-plane estimation and removal.
#
# from __future__ import annotations          # allows Plane | None on py3.9
#
# import json
# import os
# from dataclasses import dataclass
#
# import numpy as np
#
# # RANSAC defaults: a floor fills ~74% of the frame, so few iterations suffice.
# DEFAULT_ITERS = 100
# DEFAULT_TOL = 0.005         # inlier band, metres
# DEFAULT_SUBSAMPLE = 1000    # points drawn for the fit
#
#
# @dataclass
# class Plane:
#     # Plane n . p + d = 0, with n a unit normal.
#
#     n: np.ndarray           # (3,) unit normal, +z component >= 0
#     d: float                # offset, metres
#
#     def distance(self, points):
#         # Signed distance from each point to the plane, shape broadcastable.
#         return points @ self.n + self.d
#
#     def above(self, points):
#         # Signed height above the plane; >= 0 means on the normal side.
#         return points @ self.n + self.d
#
#     def to_dict(self):
#         return {"n": [float(v) for v in self.n], "d": float(self.d)}
#
#     @classmethod
#     def from_dict(cls, obj):
#         n = np.asarray(obj["n"], dtype=np.float64)
#         n = n / np.linalg.norm(n)
#         return cls(n=n, d=float(obj["d"]))
#
#
# def fit_plane_ransac(
#     points,
#     tol=DEFAULT_TOL,
#     iters=DEFAULT_ITERS,
#     subsample=DEFAULT_SUBSAMPLE,
#     seed=0,
# ):
#     # Fit the dominant plane in ``points`` (N,3) by RANSAC.
#     pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
#     if len(pts) < 3:
#         return None, 0.0
#
#     rng = np.random.default_rng(seed)
#     if subsample and len(pts) > subsample:
#         sample = pts[rng.choice(len(pts), subsample, replace=False)]
#     else:
#         sample = pts
#
#     best_inliers = 0
#     best = None
#
#     for _ in range(iters):
#         idx = rng.choice(len(sample), 3, replace=False)
#         p0, p1, p2 = sample[idx]
#
#         n = np.cross(p1 - p0, p2 - p0)
#         norm = np.linalg.norm(n)
#         if norm < 1e-9:                     # collinear triple
#             continue
#         n = n / norm
#         d = -float(n @ p0)
#
#         if n[2] < 0.0:                      # orient like a floor
#             n, d = -n, -d
#
#         inliers = int(np.sum(np.abs(sample @ n + d) < tol))
#         if inliers > best_inliers:
#             best_inliers = inliers
#             best = Plane(n=n, d=d)
#
#     if best is None:
#         return None, 0.0
#
#     return best, best_inliers / len(sample)
#
#
# def remove_plane(points, plane, tol=DEFAULT_TOL):
#     # Return (kept_mask, kept_points).
#     pts = np.asarray(points, dtype=np.float64)
#     finite = np.isfinite(pts).all(axis=-1)
#     dist = np.abs(pts @ plane.n + plane.d)
#
#     keep = finite & (dist > tol)
#     return keep, pts[keep]
#
#
# def save_plane(plane, path):
#     # Write a fitted plane to JSON.
#     payload = {
#         "version": 1,
#         "plane": plane.to_dict(),
#     }
#     with open(path, "w", encoding="utf-8") as fh:
#         json.dump(payload, fh, indent=2)
#     return path
#
#
# def load_plane(path):
#     # Read a plane from JSON. Returns None if the file is absent or invalid.
#     if not os.path.isfile(path):
#         return None
#     try:
#         with open(path, encoding="utf-8") as fh:
#             payload = json.load(fh)
#         return Plane.from_dict(payload["plane"])
#     except (OSError, ValueError, KeyError):
#         return None
#
# DEFAULT_SELF_DIST = 0.20   # camera-to-gripper cutoff, metres
#
#
# def remove_self(points, camera_pos, min_dist=DEFAULT_SELF_DIST):
#     # Drop points closer to the camera than min_dist.
#     pts = np.asarray(points, dtype=np.float64)
#     finite = np.isfinite(pts).all(axis=-1)
#     near = np.linalg.norm(pts - np.asarray(camera_pos, dtype=np.float64), axis=-1)
#     keep = finite & (near > min_dist)
#     return keep, pts[keep]
#