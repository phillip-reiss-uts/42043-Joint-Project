# Live viewer overlays for the passive viewer.
#
# The passive viewer exposes an MjvScene (viewer.user_scn) drawn on top of the
# model, plus set_texts / set_images for 2D overlays. Nothing in the project
# used them before.
#
# The hook is viewer.sync: we wrap it, so every existing sync call -- in
# Arm.move, the servo loop, the hold loops -- draws the overlay first. No
# plumbing through the call chain, and overlays appear wherever sync happens.

import numpy as np
import mujoco

from src.config import ARM_DOF

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
                 image_every=6, frustum_depth=0.16, draw_frustum=True):
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

        # pushed by the task / servo / perception paths
        self.detections = []
        self.target = None            # (p_des, R_des)
        self.info = {}                # key -> (left, right)

        self._trail = []
        self._n = 0
        self._img = None
        self._dead = set()            # sections disabled after an error
        self.control = 'open'         # 'closed' | 'held' | 'open'

        self.fovy = float(model.cam_fovy[cam_id])
        self.aspect = float(camera.width) / float(camera.height)

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
        if self.camera is None or self.image_every <= 0:
            return
        if self._n % self.image_every == 0 or self._img is None:
            # RGB pass only: the overlay has no use for depth, and this avoids
            # doubling the render cost and disturbing the depth render state.
            ren = self.camera.renderer
            ren.disable_depth_rendering()
            ren.update_scene(self.data, camera=self.camera.camera_name)
            self._img = np.array(ren.render(), copy=True)   # raw, unannotated
        if self._img is None:
            return
        vp = self.viewer.viewport
        # The rect and the array must agree exactly, so derive the rect from the
        # image aspect rather than guessing, then resample to fit.
        ih, iw = self._img.shape[:2]
        w = max(80, vp.width // 3)
        h = max(60, int(round(w * ih / float(iw))))
        if h > vp.height // 3:
            h = max(60, vp.height // 3)
            w = max(80, int(round(h * iw / float(ih))))
        ys = np.clip((np.arange(h) * ih) // h, 0, ih - 1)
        xs = np.clip((np.arange(w) * iw) // w, 0, iw - 1)
        small = np.ascontiguousarray(self._img[np.ix_(ys, xs)])
        small = self._annotate(small, sx=w / float(iw), sy=h / float(ih))
        rect = mujoco.MjrRect(vp.width - w - 8, 8, w, h)
        self.viewer.set_images([(rect, small)])

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

        tl = block(['mode', 'task', 'robot'])
        tr = block(['gripper', 'servo'])
        bl = block(['loop', 'occlusion', 'perception', 'estimate', 'observer'])
        self.viewer.set_texts([
            (FONT.mjFONTSCALE_150, GRID.mjGRID_TOPLEFT, tl[0], tl[1]),
            (FONT.mjFONTSCALE_150, GRID.mjGRID_TOPRIGHT, tr[0], tr[1]),
            (FONT.mjFONTSCALE_150, GRID.mjGRID_BOTTOMLEFT, bl[0], bl[1]),
        ])
