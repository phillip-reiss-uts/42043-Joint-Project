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

import numpy as np

from src.config import (
    SELF_FILTER, SURFACE_BAND, DEPTH_MAX, TABLE_HEIGHT,
    OBJECT_HEIGHT,
)
from src.world import ObjectWorld, TargetView


# FRAME SOURCES
class DepthSource(ABC):
    # Supplies depth + intrinsics. Deliberately not tied to RealSense, so a
    # replay file or a fake can satisfy it.

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
    # and librealsense is required), so only the interface and the Fake/Npy
    # sources have been run. Treat every call below as a claim to check on the
    # robot -- tools/record_real_frame.py does exactly that, and writes a file
    # NpyDepthSource can replay so the deprojection path is testable offline.
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
    # Renders a 30 mm cube on a table so detection has something real to find.
    # This is scaffolding for the test suite, not a substitute for hardware
    # validation (see tests/test_hardware.py --real).

    def __init__(self, width=320, height=240, cam_height=0.62, table_z=0.02,
                 object_h=0.03):
        self._w, self._h = width, height
        self.table_z = float(table_z)
        self.cam_height = float(cam_height)   # camera distance above the table
        self.object_h = float(object_h)
        # A realistic-ish D435i intrinsic at 320x240 (fx != fy, off-centre).
        self._K = np.array([[-191.3, 0.0, 160.6],
                            [0.0, 191.4, 120.9],
                            [0.0, 0.0, 1.0]])

    @property
    def K(self):
        return self._K

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
class RealObjectWorld(ObjectWorld):
    # ObjectWorld backed by a depth camera and an FK callback.
    #
    # depth_source : a DepthSource (RealSenseSource or FakeDepthSource)
    # flange_pose  : callable returning (p, R) of the gripper flange in the
    #                world/base frame -- supplied by the robot backend
    # T_flange_cam : 4x4 rigid transform from flange to camera optical frame
    #                (translation in metres). MEASURE THIS; do not assume the
    #                simulation's CAM_POS/CAM_QUAT.

    def __init__(self, depth_source, flange_pose, T_flange_cam,
                 n_objects_max=4):
        self.src = depth_source
        self._flange_pose = flange_pose
        self.T_flange_cam = np.asarray(T_flange_cam, float)
        if self.T_flange_cam.shape != (4, 4):
            raise ValueError('T_flange_cam must be 4x4')
        self.n_objects_max = int(n_objects_max)

    # BOUNDARY: the robot's own pose
    def gripper_pose(self):
        return self._flange_pose()

    # BOUNDARY: what the camera can see
    def perceive(self):
        z16, K = self.src.capture()
        points = self.deproject(z16, K)
        return self.detect(points)

    # DEPTH -> WORLD POINTS
    def camera_pose(self):
        # Camera pose in the world frame: flange pose composed with the mount.
        p_f, R_f = self._flange_pose()
        R_fc = self.T_flange_cam[:3, :3]
        t_fc = self.T_flange_cam[:3, 3]
        R_wc = np.asarray(R_f, float) @ R_fc
        p_wc = np.asarray(p_f, float) + np.asarray(R_f, float) @ t_fc
        return p_wc, R_wc

    def deproject(self, z16, K):
        # uint16 millimetres -> world-frame points above the table.
        #
        # Mirrors RGBDCamera.surface_points: the keep-mask runs in image space
        # and only survivors are lifted to 3D, because the full cloud is ~300k
        # points to keep ~1.6% of them.
        H, W = z16.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = z16.astype(np.float64) * 1e-3          # mm -> m
        p_cam_w, R_wc = self.camera_pose()

        # Ray slopes in the camera frame. camera.py's convention is x = -(u-cx)z/fx
        # and z_cam = -z; apply the same signs so the maths downstream is shared.
        ax = -(np.arange(W, dtype=np.float64) - cx) / fx
        ay = -(np.arange(H, dtype=np.float64) - cy) / fy

        near = np.sqrt(ax[None, :] ** 2 + ay[:, None] ** 2 + 1.0) * z
        wz = ((R_wc[2, 0] * ax)[None, :] + (R_wc[2, 1] * ay)[:, None]
              - R_wc[2, 2]) * z + p_cam_w[2]

        keep = (np.isfinite(z) & (z > 0.0) & (near > SELF_FILTER)
                & (near < DEPTH_MAX) & (wz > TABLE_HEIGHT + SURFACE_BAND))
        if not keep.any():
            return np.zeros((0, 3))

        vk, uk = np.nonzero(keep)
        zk = z[vk, uk]
        p_cam = np.stack([-(uk - cx) * zk / fx,
                          -(vk - cy) * zk / fy,
                          -zk], axis=-1)
        return p_cam @ R_wc.T + p_cam_w

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
            if rect is not None:
                centre_xy, yaw = rect[0], rect[1]
            else:
                centre_xy = 0.5 * (pts[:, :2].min(axis=0)
                                   + pts[:, :2].max(axis=0))
                yaw = 0.0

            centre = np.array([centre_xy[0], centre_xy[1],
                               TABLE_HEIGHT + OBJECT_HEIGHT / 2.0])
            hull = (np.column_stack([Hh, np.full(len(Hh), z_band)])
                    if Hh is not None else None)
            views.append(TargetView(centre=centre, yaw=float(yaw),
                                    span=np.asarray(span, float),
                                    score=float(score_cluster(span, len(pts))),
                                    n_points=int(len(pts)), hull=hull))

        views.sort(key=lambda v: v.score, reverse=True)
        return views[:self.n_objects_max]

    def close(self):
        self.src.close()


# VALIDATION HELPERS (used by tests/test_hardware.py --real)
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
