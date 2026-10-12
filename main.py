# vision-guided pick with UR5e + 2F-85 + D435i.
import numpy as np
import mujoco
import mujoco.viewer

# import camera
from src.camera import RGBDCamera

# import global config
from src.config import *
from src import config as cfg

# import mode loader
from src.model_loader import load_robot_model

# import perception functions
from src.perception import find_camera

# import arm functions
from src.arm import Arm

# import tasks
# pick_and_place is the OLD single-object path, no longer called by main(); it
# remains in pick_place.py for batch.py.
from src.pick_place import (ObjectRegistry, reset_scene, clear_table)

def _log(msg):
    """Optional diagnostics; gated by config.VERBOSE."""
    if cfg.VERBOSE:
        print(msg)

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
    _log(f'[INFO] {model.nbody} bodies, {model.njnt} joints, '
         f'{model.nu} actuators; wrist camera = {cam_name}')

    # leftover single-object ids -- unused by the pool/registry flow
    # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    # base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    # grip_act = model.nu - 1                       # fingers_actuator
    # free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
    rng = np.random.default_rng()

    camera = RGBDCamera(model, cam_name, width=640, height=480)
    _log(f'[INFO] intrinsics K:\n{camera.K}')

    restart = [False]
    spawn_q = [False]
    paused = [False]

    def on_key(keycode):
        if keycode in (ord('R'), ord('r')):
            restart[0] = True
        elif keycode in (ord('P'), ord('p')):
            spawn_q[0] = True
        elif keycode in (ord('E'), ord('e')):
            paused[0] = not paused[0]
            print('[INFO] e-stop %s' % ('PAUSED' if paused[0] else 'resumed'))

    def spawn_request():
        """Consume a queued spawn. True once per P press."""
        if spawn_q[0]:
            spawn_q[0] = False
            return True
        return False

    def estop():
        return paused[0]

    arm = Arm(model, data)

    if VERIFY_IK:
        arm.verify_ik()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        reg = ObjectRegistry(model, POOL_SIZE)
        print('[INFO] R = reset and clear, P = drop, E = e-stop pause, close = exit')

        def should_stop():
            """True when the run should abandon: R pressed, or the window closed."""
            return restart[0] or not viewer.is_running()

        while viewer.is_running():
            _log('[INFO] waiting -- press R to start clearing')
            while viewer.is_running() and not restart[0]:
                mujoco.mj_step(model, data)
                viewer.sync()
            if not viewer.is_running():
                break
            restart[0] = False

            reset_scene(model, data, reg, rng, viewer=viewer)
            clear_table(model, data, arm, camera, cam_id, rng, reg,
                        viewer=viewer,
                        aborted=should_stop,
                        spawn_request=spawn_request,
                        estop=estop)

            # exits on R or on the window closing, so the hold cannot outlive
            # the viewer
            while viewer.is_running() and not should_stop():
                arm.hold(GRIP_OPEN)
                mujoco.mj_step(model, data)
                viewer.sync()

            if not viewer.is_running():
                break

if __name__ == '__main__':
    main()