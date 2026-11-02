"""RealSense-backed world: the hardware implementation of the ObjectWorld seam.

WHAT THIS IS FOR
----------------
`SimObjectWorld` answers the two boundary questions -- "what can I see?" and
"where is my tool?" -- from MuJoCo. On the robot those answers come from a
RealSense and the UR controller. This module supplies the RealSense half.

The robot side does not change: src/servo.py and src/pick_place.py call
`world.perceive()` and receive `TargetView` records either way.

THE THREE THINGS THE SIMULATOR GAVE FOR FREE, AND DOES NOT HERE
---------------------------------------------------------------
1. EXTRINSICS. `data.cam_xpos` / `data.cam_xmat` come straight out of the
   physics state. On hardware the camera is bolted to the flange, so the pose
   has to be computed as FK(q) @ T_flange_camera. A few millimetres of error in
   T_flange_camera is a direct bias in every deprojected centre -- the same
   failure mode the servo's occlusion comments describe.

2. INTRINSICS. src/camera.py builds K from a single fovy, forcing fx == |fy|
   and the principal point at the exact image centre. The real D435i is close
   but not exact (fx != fy by ~0.1%, cx/cy off centre by a pixel or two). This
   module reads the device intrinsics and uses them directly. That is a
   behaviour change, not a refactor -- see check_intrinsics().

3. DEPTH CONVENTIONS. MuJoCo gives depth in metres, z-forward, and its camera
   looks down -z. RealSense gives uint16 depth in millimetres, z-forward, with
   a conventional +z-forward frame. The conversion is explicit below rather
   than hidden.

DESIGN NOTE
-----------
pyrealsense2 is imported lazily inside RealSenseSource, so importing this
module works on a machine that has no librealsense. That keeps the test suite
runnable here. `FakeDepthSource` stands in for it so the whole deprojection and
detection pipeline can still be exercised.
"""

from abc import ABC, abstractmethod
import time

import numpy as np

from src.config import (
    SELF_FILTER, SURFACE_BAND, SURFACE_Z_MIN, DEPTH_MAX, TABLE_HEIGHT,
    TABLE_XY_BOUNDS,
    OBJECT_HEIGHT, PERCEPTION_LOG_S,
)
# The depth -> world-point mask lives here, shared with the simulator's camera.
from src.camera import world_points_from_depth
from src.world import ObjectWorld, TargetView


# FRAME SOURCES
class DepthSource(ABC):
    # Supplies depth + intrinsics. Deliberately not tied to RealSense, so the
    # fake source the tests use can satisfy it.

    @abstractmethod
    def capture(self):
        # -> (depth_mm uint16 (H,W), K (3x3) in the capture's own convention).
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self):
        # (width, height)
        raise NotImplementedError

    def close(self):
        pass


class RealSenseSource(DepthSource):
    # A RealSense D435i depth stream.
    #
    # UNVERIFIED: this class has never been executed. pyrealsense2 does not
    # install on the development laptop (no wheel for Python 3.12 on macOS,
    # and librealsense is required), so only the interface and the Fake source
    # have been run. Treat every call below as a claim to check on the robot.
    #
    # The specific things to confirm first on hardware:
    #   * whether get_data() already returns millimetres (D4xx z16 usually
    #     does). If so, applying depth_scale is a 1000x error, so this class
    #     now DERIVES the factor instead of assuming it -- see capture().
    #   * get_intrinsics() returns the depth stream's, not the colour stream's;
    #     they differ on a D435i.
    #
    # Aligning depth to colour is NOT requested: perception only consumes
    # depth, and the align pass costs ~2 ms/frame. Enable rs.align if colour is
    # ever needed for the detector.

    def __init__(self, width=640, height=480, fps=30, serial=None):
        import pyrealsense2 as rs          # lazy: absent on a dev machine
        self._rs = rs
        self._w, self._h = int(width), int(height)
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial is not None:
            cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.depth, self._w, self._h,
                          rs.format.z16, fps)
        self._profile = self._pipe.start(cfg)
        self._scale = float(
            self._profile.get_device().first_depth_sensor()
            .get_depth_scale())            # metres per unit (usually 0.001)
        self._units_mm = None              # decided from the data in capture()
        self.stale_frames = 0              # frames discarded as queued/old
        self._K_cache = self._read_intrinsics()

    def _read_intrinsics(self):
        # fx, fy, cx, cy from the device. Stored in the same (fx>0, fy>0)
        # convention MuJoCo's camera.py uses for its own K, with the sign forced
        # to the sim's convention so the downstream maths is untouched.
        p = self._profile.get_stream(self._rs.stream.depth) \
            .as_video_stream_profile().get_intrinsics()
        K = np.array([[-p.fx, 0.0, p.ppx],
                      [0.0, p.fy, p.ppy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        self._intrinsics_raw = (p.fx, p.fy, p.ppx, p.ppy, p.model)
        return K

    def capture(self):
        frames = self._pipe.wait_for_frames()
        self.stale_frames = 0
        while True:
            queued = self._pipe.poll_for_frames()
            if not queued:
                break
            frames = queued
            self.stale_frames += 1
        depth = frames.get_depth_frame()
        if not depth:
            raise RuntimeError('RealSense returned no depth frame')
        arr = np.asanyarray(depth.get_data())
        z16 = self._to_millimetres(arr)
        return z16, self._K_cache

    def _to_millimetres(self, arr):
        # D4xx z16 is usually ALREADY millimetres, in which case applying
        # depth_scale again is a 1000x error. Rather than guess, decide from
        # the data: a tabletop scene is a metre or two away, so the median is
        # ~1000-3000 if already in mm, or ~1-3 if in metres.
        arr = np.asanyarray(arr)
        if arr.dtype == np.uint16:
            med = float(np.median(arr[arr > 0])) if np.any(arr > 0) else 0.0
            if med > 100.0:
                # Already millimetres.
                self._units_mm = True
                return arr
        # Raw sensor units: convert through depth_scale (metres per unit).
        self._units_mm = False
        return np.clip(np.rint(arr.astype(np.float64) * self._scale * 1000.0),
                       0, 65535).astype(np.uint16)

    @property
    def K(self):
        return self._K_cache

    @property
    def size(self):
        return self._w, self._h

    def close(self):
        try:
            self._pipe.stop()
        except Exception:
            pass

    # Report what the device actually said, for the hardware checks.
    def intrinsics_report(self):
        fx, fy, cx, cy, model = self._intrinsics_raw
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, model=str(model),
                    size=self.size)


class FakeDepthSource(DepthSource):
    # A stand-in depth source, so the pipeline is testable without librealsense.
    #
    # Renders a 60 mm cube on a table so detection has something real to find.
    # This is scaffolding for the test suite, not a substitute for hardware
    # validation (see tests/test_camera_enum.py and tests/test_camera_ruler.py).

    def __init__(self, width=320, height=240, cam_height=0.62, table_z=None,
                 object_h=None):
        self._w, self._h = width, height
        # The world frame has to put the table at the SAME height the detector
        # filters on, or every point gets discarded.
        self.table_z = float(TABLE_HEIGHT if table_z is None else table_z)
        self.cam_height = float(cam_height)   # camera distance above the table
        self.object_h = float(OBJECT_HEIGHT if object_h is None else object_h)
        # A realistic-ish D435i intrinsic at 320x240 (fx != fy, off-centre).
        self._K = np.array([[-191.3, 0.0, 160.6],
                            [0.0, 191.4, 120.9],
                            [0.0, 0.0, 1.0]])

    @property
    def K(self):
        return self._K

    @property
    def cam_world_z(self):
        # Where a world frame must place the camera for the rendered table to
        # deproject onto table_z. Fixtures should use this, not a literal.
        return self.table_z + self.cam_height

    @property
    def size(self):
        return self._w, self._h

    def capture(self):
        # Depth consistent with a table `cam_height` below the camera and a
        # cube sitting on it. Dimensions matter: the table must deproject BELOW
        # TABLE_HEIGHT so the height filter removes it, and the cube must sit
        # above it -- otherwise every pixel is "object" and clustering merges
        # the whole scene into one blob.
        table_mm = int(round(self.cam_height * 1000.0))
        top_mm = int(round((self.cam_height - self.object_h) * 1000.0))
        depth = np.full((self._h, self._w), table_mm, dtype=np.uint16)
        cy, cx = self._h // 2, self._w // 2
        # ~30 mm square: size the patch from the focal length so it deprojects
        # to a cube-sized cluster rather than an arbitrary blob.
        # The full cube is ~60 mm across (OBJECT_SIZE is a half-extent), and the
        # span test wants at least 0.70 of that. Size the patch from the FULL
        # width, not the half-width, or _span_ok rejects it as too small.
        px_per_m = abs(self._K[1, 1]) / self.cam_height
        r = max(3, int(round(0.5 * 0.060 * px_per_m)))
        depth[cy - r:cy + r, cx - r:cx + r] = top_mm
        return depth, self._K


# THE WORLD
class DepthWorld(ObjectWorld):
    # Shared hardware world: capture -> deproject -> detect.
    #
    # The camera MOUNT is the only thing a concrete world adds:
    #
    #   RealObjectWorld    eye-in-hand: camera pose = FK(q) @ T_flange_cam
    #
    # Everything downstream -- the deprojection signs, the table filter, the
    # clustering -- is shared, so the perception maths cannot drift.

    def __init__(self, depth_source, flange_pose, n_objects_max=4):
        # depth_source : a DepthSource (RealSenseSource or FakeDepthSource)
        # flange_pose  : callable returning (p, R) of the gripper base in the
        #                base frame -- supplied by the robot's kinematics
        self.src = depth_source
        self._flange_pose = flange_pose
        self.n_objects_max = int(n_objects_max)
        # SNAPSHOT CACHE. Off by default, and it must stay that way: with it off,
        # perceive() keeps nothing and the hardware path behaves exactly as it did
        # before the feature existed. run_hardware turns it on only when
        # --snapshot is given. See src/liveview.py for why this is a saved PNG
        # and not a live window.
        self.cache_frames = False
        self._last_frame = None

    def last_perception(self):
        # The most recent look, for the snapshot feature: {z16, K, p_cam_w,
        # R_wc, views}. None unless cache_frames was switched on.
        return self._last_frame

    # BOUNDARY: the robot's own pose
    def gripper_pose(self):
        return self._flange_pose()

    # BOUNDARY: what the camera can see
    def perceive(self):
        z16, K = self.src.capture()
        # One FK lookup per look at most: computed here only when the frame is
        # needed, otherwise deproject() makes the single call it always made.
        pose = self.camera_pose() if self.cache_frames else None
        points = self.deproject(z16, K, pose=pose)
        views = self.detect(points)
        if pose is not None:
            # Only the snapshot feature reads this, and only when it is on.
            self._last_frame = {'z16': z16, 'K': K, 'p_cam_w': pose[0],
                                'R_wc': pose[1], 'views': views}
        self._log_perception(len(points), views)
        return views

    # CONSOLE LOG -- the hardware world's stand-in for the overlay
    def _log_perception(self, n_points, views):
        # One line per look, because the robot has no viewer: in the simulator
        # all of this goes to the overlay panels instead.
        #
        # This is the only visibility into perception on the real cell, so it
        # reports the numbers a decision is made from -- how many points survived
        # the height mask (does the HAND FILTER actually work?), how many
        # clusters, and for the best one its centre, span and point count. A hand
        # left on the table shows up as extra points and either a second cluster
        # or a rejected span, which is exactly what there was no way to see.
        if PERCEPTION_LOG_S <= 0.0:
            return
        n_dets = len(views)
        now = time.monotonic()
        # Always report a CHANGE in detection count; otherwise rate-limit.
        changed = n_dets != getattr(self, '_log_n', None)
        if not changed and (now - getattr(self, '_log_t', 0.0)) < PERCEPTION_LOG_S:
            return
        self._log_t, self._log_n = now, n_dets
        stale = int(getattr(self.src, 'stale_frames', 0) or 0)
        if not views:
            print('[PERC] pts %6d  dets 0%s'
                  % (n_points, '  stale %d' % stale if stale else ''))
            return
        v = views[0]
        try:
            pf = np.asarray(self._flange_pose()[0], float)
            tool = '  tool (%+.3f, %+.3f, %.3f)' % (pf[0], pf[1], pf[2])
        except Exception:
            tool = ''
        print('[PERC] pts %6d  dets %d  best xy (%+.3f, %+.3f) z %.3f  '
              'span %.3f/%.3f  n %d  yaw %+.1f  score %.1f%s%s'
              % (n_points, n_dets, v.centre[0], v.centre[1], v.centre[2],
                 v.span[0], v.span[1], v.n_points, v.yaw, v.score,
                 '  stale %d' % stale if stale else '', tool))

    def camera_pose(self):
        # (p, R) of the camera in the base frame. Subclasses supply it.
        raise NotImplementedError

    # DEPTH -> WORLD POINTS
    def deproject(self, z16, K, pose=None):
        # uint16 millimetres -> world-frame points above the table.
        #
        # The mask itself lives in camera.world_points_from_depth, shared with the
        # simulator's camera, so a validity rule cannot apply to one path and not
        # the other. This method only supplies what differs here: a millimetre
        # source and a camera pose taken from the mount rather than from MuJoCo.
        #
        # `pose` is (p_cam_w, R_wc) when the caller already asked for it -- the
        # snapshot cache does -- so a look does not pay for two forward-kinematics
        # lookups. None means "ask now", which is the behaviour every existing
        # caller gets.
        p_cam_w, R_wc = self.camera_pose() if pose is None else pose
        return world_points_from_depth(
            z16.astype(np.float64) * 1e-3, K, R_wc, p_cam_w,
            SELF_FILTER, DEPTH_MAX, SURFACE_Z_MIN,
            # The same footprint the simulator uses -- one definition, so the two
            # paths cannot disagree about what counts as being on the tabletop.
            xy_bounds=TABLE_XY_BOUNDS)

    # POINTS -> TargetViews
    def detect(self, points):
        # Reuses the simulator's proven clustering/hull/calipers code. Those
        # functions are pure numpy and take no model, which is why they are
        # importable here -- see tests/test_layering.py, which keeps them that
        # way.
        from src.perception import (
            cluster_objects, _top_band, _calipers, _span_ok, score_cluster,
        )
        if len(points) < 50:
            return []

        views = []
        for pts in cluster_objects(points):
            if len(pts) < 50:
                continue
            span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
            if not _span_ok(span):
                continue

            _, Hh, z_band = _top_band(pts, 0.004)
            rect = _calipers(Hh) if Hh is not None else None
            yaw_ok = True
            if rect is not None:
                centre_xy, yaw = rect[0], rect[1]
            else:
                centre_xy = 0.5 * (pts[:, :2].min(axis=0)
                                   + pts[:, :2].max(axis=0))
                yaw = 0.0
                yaw_ok = False

            centre = np.array([centre_xy[0], centre_xy[1],
                               TABLE_HEIGHT + OBJECT_HEIGHT / 2.0])
            hull = (np.column_stack([Hh, np.full(len(Hh), z_band)])
                    if Hh is not None else None)
            views.append(TargetView(centre=centre, yaw=float(yaw),
                                    span=np.asarray(span, float),
                                    score=float(score_cluster(span, len(pts))),
                                    n_points=int(len(pts)), hull=hull,
                                    yaw_ok=bool(yaw_ok)))

        views.sort(key=lambda v: v.score, reverse=True)
        return views[:self.n_objects_max]

    def close(self):
        self.src.close()


class RealObjectWorld(DepthWorld):
    # Eye-in-hand: the camera is bolted to the wrist, so its pose is FK(q)
    # composed with the measured mount.
    #
    # T_flange_cam : 4x4 rigid transform from the gripper base to the camera's
    #                optical frame. MEASURE THIS; do not assume CAM_MOUNT.

    def __init__(self, depth_source, flange_pose, T_flange_cam,
                 n_objects_max=4):
        super().__init__(depth_source, flange_pose, n_objects_max)
        self.T_flange_cam = np.asarray(T_flange_cam, float)
        if self.T_flange_cam.shape != (4, 4):
            raise ValueError('T_flange_cam must be 4x4')

    def camera_pose(self):
        # Camera pose in the world frame: flange pose composed with the mount.
        p_f, R_f = self._flange_pose()
        R_fc = self.T_flange_cam[:3, :3]
        t_fc = self.T_flange_cam[:3, 3]
        R_wc = np.asarray(R_f, float) @ R_fc
        p_wc = np.asarray(p_f, float) + np.asarray(R_f, float) @ t_fc
        return p_wc, R_wc


# VALIDATION HELPERS (used by tests/test_camera_enum.py)
def check_intrinsics(depth_source):
    # The sim's camera.py forces fx == |fy| and a centred principal point.
    # The real device will not. Report the difference rather than hiding it.
    K = depth_source.K
    W, H = depth_source.size
    fx, fy = abs(K[0, 0]), abs(K[1, 1])
    cx, cy = K[0, 2], K[1, 2]
    return dict(
        fx=fx, fy=fy, cx=cx, cy=cy,
        fx_fy_equal=abs(fx - fy) < 1e-6,
        centred=(abs(cx - W / 2.0) < 0.5 and abs(cy - H / 2.0) < 0.5),
        centre_error_px=(cx - W / 2.0, cy - H / 2.0),
        aspect_error=abs(fx / fy - 1.0) if fy else float('nan'),
    )
