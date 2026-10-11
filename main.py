
# vision-guided pick with UR5e + 2F-85 + D435i.

import numpy as np
import mujoco
import mujoco.viewer

# import camera
from src.camera import RGBDCamera

# import global config
from src.config import *

# import mode loader
from src.model_loader import load_robot_model

# import perception functions
from src.perception import find_camera

# import arm functions
from src.arm import Arm

# import tasks
from src.pick_place import pick_and_place

def main():

    model = load_robot_model()
    
    if model is None:
        return

    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_id < 0:
        print('[ERROR] model has no "home" keyframe; cannot start.')
        return
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    cam_name, cam_id = find_camera(model)
    if cam_id < 0:
        print('[ERROR] no wrist camera found; cannot proceed.')
        return
    print(f'[INFO] {model.nbody} bodies, {model.njnt} joints, '
          f'{model.nu} actuators; wrist camera = {cam_name}')

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    grip_act = model.nu - 1                       # fingers_actuator
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
    rng = np.random.default_rng()

    camera = RGBDCamera(model, cam_name, width=640, height=480)
    print(f'[INFO] intrinsics K:\n{camera.K}')

    restart = [False]

    def on_key(keycode):
        if keycode in (ord('R'), ord('r')):
            restart[0] = True

    arm = Arm(model, data)

    if VERIFY_IK:
        arm.verify_ik()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        while viewer.is_running():
            restart[0] = False
            pick_and_place(model, data, arm, camera, cam_id, rng, viewer=viewer, aborted=lambda: restart[0])
            print('[INFO] holding -- R to re-run, close the window to exit')
            while viewer.is_running() and not restart[0]:
                arm.hold(GRIP_CLOSED)
                mujoco.mj_step(model, data)
                viewer.sync()

if __name__ == '__main__':
    main()