# Live viewer overlays for the passive viewer.
#
# The passive viewer exposes an MjvScene (viewer.user_scn) drawn on top of the
# model, plus set_texts / set_images for 2D overlays. Nothing in the project
# used them before.
#
# The hook is viewer.sync: we wrap it, so every existing sync call -- in
# Arm.move, the servo loop, the hold loops -- draws the overlay first. No
# plumbing through the call chain, and overlays appear wherever sync happens.

import time

import numpy as np
import mujoco

# DISABLED: ARM_DOF was imported but never used.
# from src.config import ARM_DOF
from src.config import (
    VIEWER_REAL_TIME, OVERLAY_CLOUD, OVERLAY_CLOUD_EVERY, SELF_FILTER,
    DEPTH_MAX, TABLE_HEIGHT, SURFACE_BAND,
)

# colour of the frustum and the trail by control regime, so the moment the
# servo stops and the arm starts driving to a saved target is visible:
#   closed = servoing on fresh observations
#   held   = servoing, but observe() is failing so the estimate is stale
#   open   = no servo; just going to a stored target
CONTROL_COLOUR = {
    'closed': {'frustum': [0.35, 0.55, 1.00, 1.0],
               'trail':   [0.05, 0.85, 1.00, 1.0]},
    'held':   {'frustum': [1.00, 0.60, 0.10, 1.0],
               'trail':   [1.00, 0.60, 0.10, 1.0]},
    'open':   {'frustum': [1.00, 0.20, 0.20, 1.0],
               'trail':   [1.00, 0.30, 0.25, 1.0]},
}

GRID = mujoco.mjtGridPos
FONT = mujoco.mjtFontScale

COLOUR = {
    'detection': np.array([0.10, 1.00, 0.25, 1.0], np.float32),
    'yaw': np.array([1.00, 0.95, 0.10, 1.0], np.float32),
    'span': np.array([0.10, 1.00, 0.25, 1.0], np.float32),
    'target': np.array([1.00, 0.25, 0.35, 1.0], np.float32),
    'trail': np.array([0.05, 0.85, 1.00, 1.0], np.float32),
    'frustum': np.array([0.35, 0.55, 1.00, 1.0], np.float32),
    'axis': np.array([1.00, 0.60, 0.10, 1.0], np.float32),
    'hull': np.array([1.00, 0.30, 1.00, 1.0], np.float32),
}

# painted onto the camera image itself (BGR order is not used; these are RGB)
PIX_GREEN = np.array([40, 255, 40], np.uint8)
PIX_YELLOW = np.array([255, 240, 40], np.uint8)
PIX_RED = np.array([255, 70, 90], np.uint8)
PIX_MAGENTA = np.array([255, 60, 255], np.uint8)


def push(viewer, key, left, right=''):
    # One-line push from anywhere that already holds the viewer handle.
    ov = getattr(viewer, 'overlay', None)
    if ov is not None:
        ov.set_info(key, left, right)


def project(p_world, cam_pos, cam_mat, K):
    # Inverse of RGBDCamera.deproject. MuJoCo cameras look down -z.
    p_cam = cam_mat.T @ (np.asarray(p_world, float) - np.asarray(cam_pos, float))
    z = -p_cam[2]
    if z <= 1e-6:
        return None
    return np.array([K[0, 2] - p_cam[0] * K[0, 0] / z,
                     K[1, 2] - p_cam[1] * K[1, 1] / z])


class Overlay:
    # Draws live robot/perception/servo state into the viewer before each sync.

    def __init__(self, viewer, model, data, camera, cam_id, arm_base_id,
                 trail_max=2600, trail_every=2, trail_draw_max=300,
                 image_every=6, frustum_depth=0.16, draw_frustum=True,
                 real_time=VIEWER_REAL_TIME, cloud=OVERLAY_CLOUD,
                 cloud_every=OVERLAY_CLOUD_EVERY, cloud_max=20000):
        self.viewer = viewer
        self.model = model
        self.data = data
        self.camera = camera
        self.cam_id = cam_id
        self.base_id = arm_base_id
        self.trail_max = trail_max
        self.trail_every = trail_every
        self.trail_draw_max = trail_draw_max
        self.image_every = image_every
        self.frustum_depth = frustum_depth
        self.draw_frustum = draw_frustum
        # Live point cloud, projected back into the camera image plane and
        # coloured by depth. Refreshed far less often than the RGB panel: it
        # needs its own depth render, which is the expensive part of the
        # pipeline, and the viewer syncs every 8 steps during a servo.
        self.cloud = bool(cloud) and camera is not None
        self.cloud_every = max(1, int(cloud_every))
        self.cloud_max = int(cloud_max)
        self._two_panels = self.cloud

        # pushed by the task / servo / perception paths
        self.detections = []
        self.target = None            # (p_des, R_des)
        self.info = {}                # key -> (left, right)

        self._trail = []
        self._n = 0
        self._img = None
        self._cloud_img = None
        self._panel_rgb = None
        self._panel_cloud = None
        self._dead = set()            # sections disabled after an error
        self.control = 'open'         # 'closed' | 'held' | 'open'

        self.fovy = float(model.cam_fovy[cam_id])
        self.aspect = float(camera.width) / float(camera.height)

        # Real-time pacing. The step loops run as fast as the renderer allows,
        # which is ~0.2x real time while waiting and ~2-3x during a servo, so
        # the two phases look nothing alike. Anchor simulation time to wall
        # time and sleep off any lead.
        self.real_time = bool(real_time)
        self._wall0 = None
        self._sim0 = 0.0

        # the hook
        self._orig_sync = viewer.sync
        viewer.sync = self._sync
        viewer.overlay = self

    # PUSHED STATE
    def set_info(self, key, left, right=''):
        self.info[key] = (str(left), str(right))

    def set_detections(self, dets):
        self.detections = list(dets) if dets else []

    def set_control(self, state):
        # 'closed' | 'held' | 'open'; drives the frustum and trail colour.
        if state in CONTROL_COLOUR:
            self.control = state

    def _col(self, kind):
        return np.array(CONTROL_COLOUR[self.control][kind], np.float32)

    def set_target(self, p_des, R_des):
        self.target = (np.array(p_des, float), np.array(R_des, float))

    def clear_target(self):
        self.target = None

    # HOOK
    def _pace(self):
        # Hold the simulation to real time. If it has fallen a long way behind
        # -- a perception stall is ~8 ms against ~0.4 ms of stepping -- re-anchor
        # instead of sprinting to catch up, because the sprint is what reads as
        # the motion lurching after every re-observation.
        if not self.real_time:
            return
        t = float(self.data.time)
        now = time.perf_counter()
        if self._wall0 is None:
            self._wall0, self._sim0 = now, t
            return
        lag = (self._wall0 + (t - self._sim0)) - now
        if lag > 0.0:
            time.sleep(min(lag, 0.05))
        elif lag < -0.05:
            self._wall0, self._sim0 = now, t

    def _sync(self, *args, **kwargs):
        try:
            self.draw()
        except Exception as exc:
            # never leave stale geometry frozen in the user scene
            try:
                self.viewer.user_scn.ngeom = 0
            except Exception:
                pass
            key = type(exc).__name__
            if key not in self._dead:
                self._dead.add(key)
                print('[OVERLAY] draw failed (%s: %s)' % (key, exc))
        self._pace()
        self._orig_sync(*args, **kwargs)

    def _section(self, fn, *args):
        # One failing section must not take the others down with it, and must
        # not leave last frame's geometry behind.
        name = fn.__name__
        if name in self._dead:
            return
        try:
            fn(*args)
        except Exception as exc:
            self._dead.add(name)
            print('[OVERLAY] %s disabled: %s: %s'
                  % (name, type(exc).__name__, exc))

    # PRIMITIVES (3D)
    def _line(self, scn, a, b, rgba, px=1.5):
        # mjGEOM_LINE size[0] is a screen-space width in PIXELS, and anything
        # below 1.0 is not drawn at all.
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, float(px),
                             np.asarray(a, np.float64),
                             np.asarray(b, np.float64))
        g.rgba[:] = np.asarray(rgba, np.float32)
        scn.ngeom += 1

    def _arrow(self, scn, a, b, rgba, radius=0.0025):
        # An ARROW geom's size[0] is a radius in METRES, not a pixel width.
        # Passing a pixel-scale number here gives a metre-scale cone.
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW1, float(radius),
                             np.asarray(a, np.float64),
                             np.asarray(b, np.float64))
        g.rgba[:] = np.asarray(rgba, np.float32)
        scn.ngeom += 1

    def _geom(self, scn, gtype, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64),
                            np.asarray(pos, np.float64),
                            np.asarray(mat, np.float64).reshape(9),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def _frame(self, scn, p, R, scale, rgba):
        for k in range(3):
            self._arrow(scn, p, p + R[:, k] * scale, rgba)

    # PRIMITIVES (2D, painted onto the camera image)
    @staticmethod
    def _px_line(img, p, q, colour, r=1):
        n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]))) + 1
        xs = np.linspace(p[0], q[0], n).round().astype(int)
        ys = np.linspace(p[1], q[1], n).round().astype(int)
        h, w = img.shape[:2]
        keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[keep], ys[keep]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                img[np.clip(ys + dy, 0, h - 1),
                    np.clip(xs + dx, 0, w - 1)] = colour

    @staticmethod
    def _px_cross(img, p, colour, r=7):
        h, w = img.shape[:2]
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < w and 0 <= y < h):
            return
        img[max(0, y - 1):y + 2, max(0, x - r):min(w, x + r + 1)] = colour
        img[max(0, y - r):min(h, y + r + 1), max(0, x - 1):x + 2] = colour

    # DRAW
    def draw(self):
        scn = self.viewer.user_scn
        scn.ngeom = 0
        self._sample()
        for fn in (self._draw_trail, self._draw_detections, self._draw_target,
                   self._draw_axes):
            self._section(fn, scn)
        if self.draw_frustum:
            self._section(self._draw_camera, scn)
        self._section(self._set_texts)
        self._section(self._set_image)
        self._section(self._set_cloud)
        # one call, so the two panels cannot disagree about the viewport
        self._section(self._push_images)

    def _sample(self):
        self._n += 1
        if self._n % self.trail_every == 0:
            # the regime is stored WITH the sample, so each segment keeps the
            # colour it was drawn in and the transition stays visible
            self._trail.append((np.array(self.data.xpos[self.base_id], float),
                                self.control))
            if len(self._trail) > self.trail_max:
                self._trail.pop(0)

    def _draw_trail(self, scn):
        n = len(self._trail)
        if n < 2:
            return
        # A full pick is mostly open-loop, so the trail has to span long enough
        # to still hold the servo phase. Draw at most trail_draw_max segments,
        # subsampled uniformly, to bound the per-frame geom cost.
        stride = max(1, n // self.trail_draw_max)
        idxs = list(range(0, n - stride, stride))
        for i in idxs:
            a, _ = self._trail[i]
            b, cb = self._trail[min(i + stride, n - 1)]
            col = np.array(CONTROL_COLOUR[cb]['trail'], np.float32)
            col[3] = 0.35 + 0.65 * (i / float(n - 1))
            self._line(scn, a, b, col, px=1.5)

    def _draw_detections(self, scn):
        for d in self.detections:
            c = np.array(d.centre, float)
            self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                       [0.008, 0.008, 0.008], c, np.eye(3).flatten(),
                       COLOUR['detection'])
            yaw = np.radians(float(getattr(d, 'yaw', 0.0)))
            span = getattr(d, 'span', None)
            half = 0.5 * (float(span[0]) if span is not None else 0.03)
            axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            self._arrow(scn, c - axis * half, c + axis * half, COLOUR['yaw'],
                        radius=0.003)
            if span is not None:
                size = np.array([span[0] / 2.0, span[1] / 2.0, 0.006])
                self._geom(scn, mujoco.mjtGeom.mjGEOM_LINEBOX, size, c,
                           np.eye(3).flatten(), COLOUR['span'])
            # the top-face hull the yaw was fitted to
            hull = getattr(d, 'hull', None)
            if hull is not None and len(hull) >= 3:
                for i in range(len(hull)):
                    self._line(scn, hull[i], hull[(i + 1) % len(hull)],
                               COLOUR['hull'], px=2.0)
            self._line(scn, c, c + np.array([0.0, 0.0, 0.05]),
                       COLOUR['detection'], px=1.5)

    def _draw_target(self, scn):
        if self.target is None:
            return
        p, R = self.target
        self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                   [0.006, 0.006, 0.006], p, np.eye(3).flatten(),
                   COLOUR['target'])
        self._frame(scn, p, R, 0.05, COLOUR['target'])

    def _draw_camera(self, scn):
        p = np.array(self.data.cam_xpos[self.cam_id], float)
        R = np.array(self.data.cam_xmat[self.cam_id], float).reshape(3, 3)
        d = self.frustum_depth
        hh = d * np.tan(np.radians(self.fovy) / 2.0)
        hw = hh * self.aspect
        corners = [(-hw, +hh), (+hw, +hh), (+hw, -hh), (-hw, -hh)]
        world = [p + R @ np.array([cx, cy, -d]) for cx, cy in corners]
        col = self._col('frustum')
        for w in world:
            self._line(scn, p, w, col, px=1.2)
        for i in range(4):
            self._line(scn, world[i], world[(i + 1) % 4], col, px=1.2)
        self._arrow(scn, p, p + R @ np.array([0.0, 0.0, -d]), col,
                    radius=0.004)

    def _draw_axes(self, scn):
        o = np.zeros(3)
        self._line(scn, o, o + np.array([0.10, 0, 0]), COLOUR['axis'], px=1.5)
        self._line(scn, o, o + np.array([0, 0.10, 0]), COLOUR['axis'], px=1.5)
        self._line(scn, o, o + np.array([0, 0, 0.10]), COLOUR['axis'], px=1.5)

    # CAMERA IMAGE, ANNOTATED
    def _annotate(self, img, sx=1.0, sy=1.0):
        # Paint what the detector concluded onto the frame it concluded it
        # from: the fitted centre, the yaw axis and the measured span, all
        # projected back into image space with the same intrinsics and
        # extrinsics the deprojection used.
        #
        # Called AFTER any downscale, with sx/sy the scale factors, so the mark
        # widths are in final pixels. Annotating at full resolution and then
        # subsampling leaves lines a pixel wide and patchy.
        if img.ndim != 3 or img.shape[2] < 3:
            return img
        p = np.array(self.data.cam_xpos[self.cam_id], float)
        R = np.array(self.data.cam_xmat[self.cam_id], float).reshape(3, 3)
        K = self.camera.K

        def proj(q):
            uv = project(q, p, R, K)
            return None if uv is None else np.array([uv[0] * sx, uv[1] * sy])

        for d in self.detections:
            c = np.array(d.centre, float)
            span = getattr(d, 'span', None)
            if span is not None:
                hx, hy = float(span[0]) / 2.0, float(span[1]) / 2.0
                corners = [np.array([c[0] - hx, c[1] - hy, c[2]]),
                           np.array([c[0] + hx, c[1] - hy, c[2]]),
                           np.array([c[0] + hx, c[1] + hy, c[2]]),
                           np.array([c[0] - hx, c[1] + hy, c[2]])]
                pxs = [proj(q) for q in corners]
                if all(q is not None for q in pxs):
                    for i in range(4):
                        self._px_line(img, pxs[i], pxs[(i + 1) % 4], PIX_GREEN)
            yaw = np.radians(float(getattr(d, 'yaw', 0.0)))
            axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            # the top-face hull the yaw was fitted to, drawn as its edges
            hull = getattr(d, 'hull', None)
            if hull is not None and len(hull) >= 3:
                hp = [proj(q) for q in hull]
                for i in range(len(hp)):
                    a, b = hp[i], hp[(i + 1) % len(hp)]
                    if a is not None and b is not None:
                        self._px_line(img, a, b, PIX_MAGENTA, r=1)
            a = proj(c - axis * 0.045)
            b = proj(c + axis * 0.045)
            if a is not None and b is not None:
                self._px_line(img, a, b, PIX_YELLOW, r=1)
            q = proj(c)
            if q is not None:
                self._px_cross(img, q, PIX_RED)

        if self.target is not None:
            q = proj(self.target[0])
            if q is not None:
                self._px_cross(img, q, PIX_GREEN, r=11)
        return img

    def _set_image(self):
        self._panel_rgb = None
        if self.camera is None or self.image_every <= 0:
            return
        if self._n % self.image_every == 0 or self._img is None:
            # RGB pass only: the RGB panel has no use for depth, and this avoids
            # doubling its render cost. _set_cloud runs its own depth pass.
            ren = self.camera.renderer
            ren.disable_depth_rendering()
            ren.update_scene(self.data, camera=self.camera.camera_name)
            self._img = np.array(ren.render(), copy=True)   # raw, unannotated
        if self._img is None:
            return
        # The rect and the array must agree exactly, so derive the rect from the
        # image aspect rather than guessing, then resample to fit.
        ih, iw = self._img.shape[:2]
        got = self._panel_rect('rgb')
        if got is None:
            return
        rect, w, h = got
        ys = np.clip((np.arange(h) * ih) // h, 0, ih - 1)
        xs = np.clip((np.arange(w) * iw) // w, 0, iw - 1)
        small = np.ascontiguousarray(self._img[np.ix_(ys, xs)])
        small = self._annotate(small, sx=w / float(iw), sy=h / float(ih))
        self._panel_rgb = (rect, small)

    # POINT CLOUD PANEL
    @staticmethod
    def _depth_ramp(t):
        # near = blue, far = red. Written out rather than pulled from matplotlib,
        # which the viewer has no other reason to import.
        t = np.clip(np.asarray(t, float), 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1) * 255.0

    def _set_cloud(self):
        # The live object point cloud, drawn as the camera sees it: every point
        # the depth pipeline kept, projected back into the image plane and
        # coloured by range. This is the one view that shows the sensor's own
        # data rather than only what perception concluded from it.
        if not self.cloud or self.camera is None:
            return
        self._panel_cloud = None
        if self._cloud_img is None or self._n % self.cloud_every == 0:
            depth = self.camera.capture_depth(self.data)
            # z_min=0 keeps the table as well, so the cloud is shown in context
            # and the height filter that removes it is visible as a colour
            # change. surface_points restores nothing, so leave depth rendering
            # as it finds it -- _set_image disables it before its own render.
            pts = self.camera.surface_points(
                depth, self.data, SELF_FILTER, DEPTH_MAX, 0.0)
            self._cloud_img = self._splat(pts)

        img = self._cloud_img
        if img is None:
            return
        uvl = self._panel_rect('cloud')
        if uvl is None:
            return
        rect, w, h = uvl
        ih, iw = img.shape[:2]
        ys = np.clip((np.arange(h) * ih) // h, 0, ih - 1)
        xs = np.clip((np.arange(w) * iw) // w, 0, iw - 1)
        self._panel_cloud = (rect, np.ascontiguousarray(img[np.ix_(ys, xs)]))

    def _splat(self, pts):
        # Scatter world points into a camera-resolution image, coloured by range.
        H, W = int(self.camera.height), int(self.camera.width)
        img = np.zeros((H, W, 3), np.float64)
        if pts is None or len(pts) == 0:
            return img.astype(np.uint8)

        if len(pts) > self.cloud_max:
            pts = pts[::max(1, len(pts) // self.cloud_max)]

        cam_pos = np.array(self.data.cam_xpos[self.cam_id], float)
        cam_mat = np.array(self.data.cam_xmat[self.cam_id],
                           float).reshape(3, 3)
        pc = (pts - cam_pos) @ cam_mat          # camera frame
        z = -pc[:, 2]
        ok = z > 1e-6
        if not ok.any():
            return img.astype(np.uint8)
        pc, z = pc[ok], z[ok]

        u = np.rint(self.camera.K[0, 2] - pc[:, 0] * self.camera.K[0, 0] / z)
        v = np.rint(self.camera.K[1, 2] - pc[:, 1] * self.camera.K[1, 1] / z)
        u = u.astype(np.int64)
        v = v.astype(np.int64)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[inside], v[inside], z[inside]
        if len(z) == 0:
            return img.astype(np.uint8)

        # The table gets a flat neutral grey: it is context, and it outnumbers
        # the object roughly 40:1, so letting it drive the colour map left the
        # whole panel a uniform dark red. The object then gets the full ramp
        # over its own depth range, with the ends clipped so a single stray
        # point cannot flatten the spread.
        world = pts[ok][inside]
        is_table = world[:, 2] <= TABLE_HEIGHT + SURFACE_BAND
        img[v[is_table], u[is_table]] = (70.0, 70.0, 70.0)
        zo = z[~is_table]
        if len(zo):
            lo, hi = np.percentile(zo, 5.0), np.percentile(zo, 95.0)
            img[v[~is_table], u[~is_table]] = self._depth_ramp(
                (zo - lo) / max(float(hi - lo), 1e-6))
        return img.astype(np.uint8)

    def _panel_rect(self, slot='rgb'):
        # The camera image sits at the right edge; the cloud sits immediately to
        # its left when shown. Each slot needs its own origin -- sharing one
        # origin put both panels on top of each other.
        vp = self.viewer.viewport
        w = max(80, vp.width // 3)
        h = max(60, int(round(w * self.camera.height
                              / float(self.camera.width))))
        if h > vp.height // 3:
            h = max(60, vp.height // 3)
            w = max(80, int(round(h * self.camera.width
                                  / float(self.camera.height))))
        if slot == 'cloud' and self._two_panels:
            left = vp.width - (2 * w + 16)
        else:
            left = vp.width - w - 8
        return mujoco.MjrRect(left, 8, w, h), w, h

    def _push_images(self):
        panels = [p for p in (self._panel_rgb, self._panel_cloud) if p]
        if panels:
            self.viewer.set_images(panels)

    # TEXT
    def _auto_lines(self):
        d, m = self.data, self.model
        p = np.array(d.xpos[self.base_id], float)
        cp = np.array(d.cam_xpos[self.cam_id], float)
        grip = float(d.ctrl[m.nu - 1]) if m.nu else 0.0
        return {
            'robot': ('t %.2f s   step %d   pos %4.2f %4.2f %4.2f'
                      % (d.time, self._n, p[0], p[1], p[2]),
                      'contacts %d' % d.ncon),
            'gripper': ('grip cmd %5.1f / 255' % grip,
                        'cam %4.2f %4.2f %4.2f' % (cp[0], cp[1], cp[2])),
            # default, overridden by whatever perceive_all last reported
            'perception': ('dets %d' % len(self.detections),
                           'awaiting detection'),
            'loop': ('loop: %s' % self.control,
                     'closed=fresh  held=stale  open=saved target'),
        }

    def _set_texts(self):
        info = self._auto_lines()
        info.update(self.info)

        def block(keys):
            left, right = [], []
            for k in keys:
                if k in info:
                    left.append(str(info[k][0]))
                    right.append(str(info[k][1]) if len(info[k]) > 1 else '')
            return ('\n'.join(left), '\n'.join(right))

        tl = block(['mode', 'task', 'move', 'robot'])
        tr = block(['gripper', 'servo'])
        bl = block(['loop', 'occlusion', 'perception', 'estimate', 'observer'])
        self.viewer.set_texts([
            (FONT.mjFONTSCALE_150, GRID.mjGRID_TOPLEFT, tl[0], tl[1]),
            (FONT.mjFONTSCALE_150, GRID.mjGRID_TOPRIGHT, tr[0], tr[1]),
            (FONT.mjFONTSCALE_150, GRID.mjGRID_BOTTOMLEFT, bl[0], bl[1]),
        ])
