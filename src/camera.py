import math
import numpy as np

# MuJoCo is imported INSIDE the functions that need it, not here.
#
# This module is split at that line: `world_points_from_depth` is pure numpy and
# is the ONE depth validity rule, shared by the simulated camera and by
# `world_real`'s RealSense path. `world_real` therefore imports this module, and
# a module-level `import mujoco` here dragged the simulator into the HARDWARE
# front-end -- `import run_hardware` pulled it in, which is exactly what
# run_hardware's own docstring says it does not do, on a laptop that runs the
# arm without the simulator installed.
#
# This is the same shape `src.state_space` uses for its default provider: the
# dependency is real but is resolved at call time, so importing the module needs
# no simulator. tests/test_ur_backend.py asserts the front-end stays clean.


def world_points_from_depth(z, K, R_wc, p_cam_w, near_min, near_max, z_min,
                            xy_bounds=None):
    # THE ONE implementation of "which depth pixels are valid, and where they are
    # in the base frame".
    #
    # Both front ends call this: RGBDCamera.surface_points (a MuJoCo camera in
    # simulation) and DepthWorld.deproject (a RealSense or fake source on
    # hardware). They used to carry separate copies of the same maths, which is
    # exactly how a validity rule added to one path silently failed to reach the
    # other -- the robot base was filtered out of the hardware detections and
    # still clustered in the simulation.
    #
    # z is depth in METRES. R_wc / p_cam_w are the camera pose in the base frame.
    # xy_bounds, when given, is (centre_x, centre_y, half_x, half_y): the only
    # region where a graspable object can be -- the measured tabletop. Anything
    # outside it is the base, the column, the arm, the bench edge or the floor.
    #
    # Masking runs on (H,W) arrays and only survivors are lifted to 3D: the full
    # cloud is ~300k points to keep ~1.6% of them.
    H, W = z.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1-D ray slopes, broadcast against depth only where a full (H,W) plane is
    # genuinely needed. meshgrid here costs several 2.4 MB temporaries.
    ax = -(np.arange(W, dtype=np.float64) - cx) / fx
    ay = -(np.arange(H, dtype=np.float64) - cy) / fy

    # |p_cam| = z * |(ax, ay, -1)|, and R is orthonormal so this is also the
    # distance from the camera in world coordinates
    near = np.sqrt(ax[None, :] ** 2 + ay[:, None] ** 2 + 1.0) * z
    # world coordinates are linear in depth, so no 3D cloud is needed for the tests
    wx = ((R_wc[0, 0] * ax)[None, :] + (R_wc[0, 1] * ay)[:, None]
          - R_wc[0, 2]) * z + p_cam_w[0]
    wy = ((R_wc[1, 0] * ax)[None, :] + (R_wc[1, 1] * ay)[:, None]
          - R_wc[1, 2]) * z + p_cam_w[1]
    wz = ((R_wc[2, 0] * ax)[None, :] + (R_wc[2, 1] * ay)[:, None]
          - R_wc[2, 2]) * z + p_cam_w[2]

    keep = (np.isfinite(z) & (z > 0.0) & (near > near_min)
            & (near < near_max) & (wz > z_min))
    if xy_bounds is not None:
        bx, by, hx, hy = xy_bounds
        keep &= (np.abs(wx - bx) <= hx) & (np.abs(wy - by) <= hy)
    if not keep.any():
        return np.zeros((0, 3))

    # rows are v, cols are u
    vk, uk = np.nonzero(keep)
    zk = z[vk, uk]
    p_cam = np.stack([-(uk - cx) * zk / fx,
                      -(vk - cy) * zk / fy,
                      -zk], axis=-1)
    return p_cam @ R_wc.T + p_cam_w


class RGBDCamera:
    # Wraps a MuJoCo camera as an RGB-D sensor.

    def __init__(self, model, camera_name, width=640, height=480):
        import mujoco                 # see the module header
        self.model = model
        self.camera_name = camera_name
        self.width = width
        self.height = height

        # Resolve camera id once
        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")

        # Renderer (GPU context, allocated once)
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # Intrinsics (constant for this camera + resolution)
        self.K = self._compute_intrinsics()

    # Intrinsics
    def _compute_intrinsics(self):
        # Pinhole intrinsics from MuJoCo's cam_fovy.
        fovy = math.radians(self.model.cam_fovy[self.cam_id])
        f = 0.5 * self.height / math.tan(fovy / 2.0)

        K = np.array([
            [-f,  0.0, self.width  / 2.0],
            [ 0.0,  f, self.height / 2.0],
            [ 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K

    # Capture
    def capture_depth(self, data):
        # Depth pass only. The point pipeline never uses the RGB frame, and the
        # extra render costs ~8 ms inside the servo loop.
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        return self.renderer.render()

    def capture(self, data):
        # Render the current view from this camera.
        # RGB pass
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render()

        # Depth pass
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        depth = self.renderer.render()

        return rgb, depth

    # Deprojection
    def deproject(self, depth, data):
        # Convert a dense depth map to a dense world-frame point cloud.

        import mujoco                 # see the module header
        mujoco.mj_forward(self.model, data)

        H, W = depth.shape
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Pixel grid
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)  # (H, W)

        # Validity mask
        valid = np.isfinite(depth) & (depth > 0.0)

        # Camera-frame points (dense)
        z = depth.astype(np.float64)
        #x_cam = (uu - cx) * z / fx
        x_cam = -(uu - cx) * z / fx
        #y_cam = (vv - cy) * z / fy
        y_cam = -(vv - cy) * z / fy      
        z_cam = -z

        # Stack into (H, W, 3), mark invalid as NaN
        p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        p_cam[~valid] = np.nan

        # Camera extrinsics from current sim state
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        t = data.cam_xpos[self.cam_id]

        # To world frame: p_world = R @ p_cam + t
        flat = p_cam.reshape(-1, 3)
        world = flat @ R.T + t
        points = world.reshape(H, W, 3)

        return points

    # Deprojection, masked in image space
    def surface_points(self, depth, data, near_min, near_max, z_min,
                       xy_bounds=None):
        # Thin wrapper: this class knows where the MuJoCo camera is, and
        # world_points_from_depth owns the validity rule. See that function for
        # why there is only one copy of the mask.
        import mujoco                 # see the module header
        mujoco.mj_forward(self.model, data)
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        t = data.cam_xpos[self.cam_id]
        return world_points_from_depth(np.asarray(depth, dtype=np.float64),
                                       self.K, R, t,
                                       near_min, near_max, z_min, xy_bounds)

def find_camera(model):
    # Return (name, id) of the wrist camera, or (None, -1).
    #
    # Simulation-only: it resolves a camera in a MuJoCo model. Kept here rather
    # than in perception so the perception geometry (clustering, calipers,
    # yaw) stays importable on a machine with no MuJoCo -- which is what the
    # hardware front-end needs.
    import mujoco                       # see the module header
    for candidate in ('camera/depth_cam', 'depth_cam'):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
        if cid >= 0:
            return candidate, cid
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and 'depth_cam' in name:
            return name, i
    return None, -1
