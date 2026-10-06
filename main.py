import numpy as np
import mujoco
import mujoco.viewer

from src.model_loader import load_robot_model
from src.camera import RGBDCamera


def main():
    model = load_robot_model()
    if model is None:
        return

    data = mujoco.MjData(model)

    # Reset to home keyframe if available
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        print("[INFO] Reset to 'home' keyframe.")
    else:
        print("[INFO] No 'home' keyframe found, using default state.")

    # Model summary
    print(f"[INFO] Model loaded: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.nu} actuators, {model.ncam} cameras")
    print("[INFO] Cameras in model:")
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"       Camera {i}: {cam_name}")

    # Locate the wrist camera in the compiled model
    cam_name = "camera/depth_cam"
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        cam_name = "depth_cam"
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name and "depth_cam" in name:
                cam_name = name
                cam_id = i
                break
    if cam_id < 0:
        print("[ERROR] No wrist camera found; cannot proceed.")
        return
    print(f"[INFO] Using wrist camera: {cam_name}")

    # Build the RGB-D camera wrapper
    camera = RGBDCamera(model, cam_name, width=640, height=480)

    print(f"[INFO] Intrinsics K:\n{camera.K}")

    # Viewer
    print("[INFO] Launching viewer. Press ESC to exit.")
    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Enable camera frustum visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        
        # Show the wrist camera view
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        while viewer.is_running():
            mujoco.mj_step(model, data)
            step += 1

            # Render RGB-D every 10 physics steps (~20 Hz at 500 Hz sim)
            if step % 10 == 0:
                rgb, depth = camera.capture(data)
                points = camera.deproject(depth, data)

                valid = np.isfinite(points).all(axis=-1)
                n_valid = int(valid.sum())
                if n_valid > 0:
                    z = depth[np.isfinite(depth) & (depth > 0)]
                    print(f"[RGB-D] rgb={rgb.shape} depth={depth.shape} "
                          f"valid={n_valid} "
                          f"z=[{z.min():.3f}, {z.max():.3f}] m")

            viewer.sync()


if __name__ == "__main__":
    main()