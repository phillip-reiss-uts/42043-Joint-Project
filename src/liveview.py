"""Annotated depth snapshots for the hardware run.

WHY THIS EXISTS
---------------
The hardware front-end has no viewer: `push()` finds no overlay and does nothing,
so the only feedback from perception is the `[PERC]` console line. That line gives
numbers but cannot show WHY a look failed -- whether an operator's hand is in the
cloud, whether the jaws are clipping the cube, whether the depth is simply wrong.

This writes one PNG on demand. It is deliberately NOT a live window:

    * a live `cv2` window needs a display, and this program runs over SSH;
    * a draw inside the servo loop costs 1-3 ms per iteration against an 8 ms
      command period, which starves the servoj stream and stops the arm;
    * the frames that matter are the ones perception already took, so nothing has
      to be re-captured and no second consumer touches the camera.

Callers hold the DECISION, this module only draws. Nothing here imports mujoco --
it is reachable from the hardware path, and tests/test_layering keeps it that way.

PROJECTION
----------
The exact inverse of camera.world_points_from_depth, which is the one place the
depth -> world maths lives:

    ray = R_wc.T @ (p_world - p_cam_w)      # camera frame
    z   = -ray[2]                           # depth along the optical axis
    ax, ay = ray[0] / z, ray[1] / z
    u = K[0,2] - ax * K[0,0]
    v = K[1,2] - ay * K[1,1]

Note the sign of K[0,0]: RealSenseSource stores a NEGATIVE fx to keep the sim's
convention, so the signs in the two lines above are not a typo. tests/test_liveview
round-trips this against world_points_from_depth rather than trusting it.
"""

import os
import time

import numpy as np


# ---- projection -----------------------------------------------------------
def project(p_world, p_cam_w, R_wc, K):
    # (u, v) pixel of a world point, or None if it is behind the camera.
    p = np.asarray(p_world, float) - np.asarray(p_cam_w, float)
    ray = np.asarray(R_wc, float).T @ p
    z = -ray[2]
    if z <= 1e-6:
        return None
    ax, ay = ray[0] / z, ray[1] / z
    u = float(K[0, 2] - ax * K[0, 0])
    v = float(K[1, 2] - ay * K[1, 1])
    return u, v


# ---- drawing --------------------------------------------------------------
def depth_image(z16, near_m=0.15, far_m=1.60):
    # False-colour depth. Near is bright, far is dark, and anything with no
    # return is black so a dropout is obvious rather than blending in.
    import cv2

    z = np.asarray(z16, float) / 1000.0
    valid = z > 0.0
    t = np.clip((z - near_m) / max(far_m - near_m, 1e-6), 0.0, 1.0)
    grey = np.where(valid, (1.0 - t) * 255.0, 0.0).astype(np.uint8)
    img = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)
    img[~valid] = (0, 0, 0)
    return img


def annotate(z16, K, p_cam_w, R_wc, views=(), note='', tool_pose=None):
    # Draw the depth frame with the detections on it. Returns a BGR image.
    #
    # `views` are TargetView records from the world boundary, so this draws
    # exactly what perception decided -- not a re-derivation of it.
    import cv2

    img = depth_image(z16)

    for i, v in enumerate(views):
        # hull first, so the centre marker sits on top of it
        hull = getattr(v, 'hull', None)
        if hull is not None and len(hull) >= 3:
            pts = []
            for p in np.asarray(hull, float):
                uv = project(p, p_cam_w, R_wc, K)
                if uv is not None:
                    pts.append(uv)
            if len(pts) >= 3:
                poly = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [poly], True, (0, 255, 0), 2)

        uv = project(np.asarray(v.centre, float), p_cam_w, R_wc, K)
        if uv is not None:
            u, w = int(round(uv[0])), int(round(uv[1]))
            cv2.drawMarker(img, (u, w), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(img, '%d %.3f %.3f' % (i, v.centre[0], v.centre[1]),
                        (u + 12, w - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1, cv2.LINE_AA)

    if tool_pose is not None:
        uv = project(np.asarray(tool_pose, float), p_cam_w, R_wc, K)
        if uv is not None:
            cv2.drawMarker(img, (int(round(uv[0])), int(round(uv[1]))),
                           (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 1)

    head = '%s  depth mm: min %d max %d'
    z = np.asarray(z16)
    zpos = z[z > 0]
    lo = int(zpos.min()) if zpos.size else 0
    hi = int(zpos.max()) if zpos.size else 0
    cv2.putText(img, head % (time.strftime('%H:%M:%S'), lo, hi),
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    if note:
        cv2.putText(img, note, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ---- saving ---------------------------------------------------------------
class Snapshotter:
    # Writes snap_NNN.png into a directory. Only ever called between policy
    # steps, so it cannot delay a control command.
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.n = 0
        self.last = None

    def save(self, img, prefix='snap'):
        self.n += 1
        path = os.path.join(self.outdir, '%s_%03d.png' % (prefix, self.n))
        import cv2
        if not cv2.imwrite(path, img):
            raise RuntimeError('could not write %s' % path)
        self.last = path
        return path
