import math
import numpy as np
import mujoco


class RGBDCamera:
    """
    Wraps a MuJoCo camera as an RGB-D sensor.

    Produces dense RGB and depth images, computes pinhole intrinsics
    from the model's fovy, and deprojects depth to a dense world-frame
    point cloud.

    Filtering (plane removal, clustering, validity masks) is deliberately
    left to the caller — this class reports what the sensor sees.
    """

    def __init__(self, model, camera_name, width=640, height=480):
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

    # ------------------------------------------------------------------
    # Intrinsics
    # ------------------------------------------------------------------
    def _compute_intrinsics(self):
        """
        Pinhole intrinsics from MuJoCo's cam_fovy.

        Note: the first element (fx) is negated. MuJoCo's depth
        deprojection convention requires this; without it, points
        mirror horizontally in the camera frame.
        """
        fovy = math.radians(self.model.cam_fovy[self.cam_id])
        f = 0.5 * self.height / math.tan(fovy / 2.0)

        K = np.array([
            [-f,  0.0, self.width  / 2.0],
            [ 0.0,  f, self.height / 2.0],
            [ 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return K

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture(self, data):
        """
        Render the current view from this camera.

        Returns
        -------
        rgb   : (H, W, 3) uint8
        depth : (H, W)    float32, metric distance in meters
        """
        # RGB pass
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render()

        # Depth pass
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        depth = self.renderer.render()

        return rgb, depth

    # ------------------------------------------------------------------
    # Deprojection
    # ------------------------------------------------------------------
    def deproject(self, depth, data):
        """
        Convert a dense depth map to a dense world-frame point cloud.

        Parameters
        ----------
        depth : (H, W) float32
        data  : MjData, used for current camera extrinsics

        Returns
        -------
        points : (H, W, 3) float64 — one world-frame point per pixel.
                 Invalid pixels (depth <= 0 or non-finite) are set to NaN.
        """

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
        # Apply to flattened points, then reshape back
        flat = p_cam.reshape(-1, 3)
        world = flat @ R.T + t
        points = world.reshape(H, W, 3)

        return points