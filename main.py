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
from src.pick_place import (ObjectRegistry, reset_scene, clear_table,
                            place_cube, put_cube_down)


def _log(msg):
    # Optional diagnostics; gated by config.VERBOSE.
    if cfg.VERBOSE:
        print(msg)


# MODES
WAIT = 'WAIT'
SEARCH = 'SEARCH'


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

    home_st = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home_st, home_id)
    home_q = np.array(home_st.qpos[:ARM_DOF])

    cam_name, cam_id = find_camera(model)
    if cam_id < 0:
        print('[ERROR] no wrist camera found; cannot proceed.')
        return
    _log(f'[INFO] {model.nbody} bodies, {model.njnt} joints, '
         f'{model.nu} actuators; wrist camera = {cam_name}')

    rng = np.random.default_rng()

    camera = RGBDCamera(model, cam_name, width=640, height=480)
    _log(f'[INFO] intrinsics K:\n{camera.K}')

    # COMMANDS
    # Keys 6-9 only: the viewer maps a-z onto its 26 render flags (S toggles
    # shadows, E constraints, P contact-split, R reflection) and 0-5 onto geom
    # groups, so no letter or low digit is available to us.
    cmd = {'start': False, 'pause': False, 'place': False, 'reset': False}

    def on_key(keycode):
        if keycode == ord('6'):
            cmd['place'] = True
        elif keycode == ord('7'):
            cmd['start'] = True
        elif keycode == ord('8'):
            cmd['pause'] = True
        elif keycode == ord('9'):
            cmd['reset'] = True

    def take(name):
        # Consume a flag: True once per press.
        if cmd[name]:
            cmd[name] = False
            return True
        return False

    def pending():
        # Non-consuming peek, safe to call every step.
        return cmd['reset'] or cmd['place'] or cmd['start']

    arm = Arm(model, data)

    if VERIFY_IK:
        arm.verify_ik()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        reg = ObjectRegistry(model, POOL_SIZE)
        reset_scene(model, data, reg, rng, viewer=None)

        print('[INFO] 6 = spawn cube, 7 = start/stop search, 8 = pause, '
              '9 = reset scene')
        print('[STATE] WAIT  (scene empty; press 6 to spawn the cube, '
              '7 to search)')

        paused = [False]
        mode = [WAIT]

        def estop():
            # Consume the pause command here, not at the outer-loop boundary:
            # that boundary is not reached while clear_table is running, so a
            # mid-search press used to queue up and fire late. Consuming it per
            # step also stops two quick presses collapsing into one toggle.
            if cmd['pause']:
                cmd['pause'] = False
                paused[0] = not paused[0]
                say('PAUSED  (holding position)' if paused[0] else 'RESUMED')
            return paused[0]

        def say(msg):
            print('[STATE] %s' % msg)

        def at_home():
            return float(np.linalg.norm(data.qpos[:ARM_DOF] - home_q)) < 1e-2

        def enter_wait(reason):
            mode[0] = WAIT
            say('WAIT  (%s)' % reason)
            if not at_home():
                say('not at home -- returning home')
                arm.move(home_q, GRIP_OPEN, rate=RAPID, viewer=viewer,
                         estop=estop)
                say('WAIT  (at home)')

        def hard_reset():
            # Instant: the arm is teleported, not driven. Every parameter is
            # cleared, including the pause flag and the command queue.
            mujoco.mj_resetDataKeyframe(model, data, home_id)
            mujoco.mj_forward(model, data)
            for i in range(len(reg.body_ids)):
                reg.teleport(data, i, PARK_POS)
                reg.state[i] = 'parked'
            mujoco.mj_forward(model, data)
            for k in cmd:
                cmd[k] = False
            paused[0] = False
            mode[0] = WAIT
            say('WAIT  (9 reset: cube parked, arm at home, all state cleared)')

        while viewer.is_running():

            # RESET pre-empts everything and needs no motion
            if take('reset'):
                hard_reset()
                viewer.sync()
                continue

            # PAUSE
            if take('pause'):
                paused[0] = not paused[0]
                if paused[0]:
                    say('PAUSED  (holding position)')
                else:
                    say('RESUMED')

            if mode[0] == WAIT:
                # P places the cube; the robot stays in WAIT
                if take('place'):
                    say('SPAWN  (6)')
                    idx = place_cube(model, data, reg, rng, viewer=viewer,
                                     stop=lambda: cmd['reset'], estop=estop)
                    say('WAIT  (cube placed on the table, index %s)' % idx)
                # 7 starts a search
                if take('start'):
                    if not reg.active():
                        say('SEARCH ignored -- no cube on the table '
                            '(press 6 first)')
                    else:
                        mode[0] = SEARCH
                arm.hold(GRIP_OPEN)
                mujoco.mj_step(model, data)
                viewer.sync()
                continue

            # SEARCH
            say('SEARCH  (7)')

            if USE_OBSERVER:
                # imported here so scipy is not loaded when the observer is off
                from src.observer import ArmObserver
                from src.controller import (
                    JointController, LQRController, Encoder, lqr_weights,
                )
                observer = ArmObserver(
                    model, data, data.qpos[:ARM_DOF],
                    lqr_weights=(lqr_weights() if USE_LQR else None))
                ctl = (LQRController(observer) if USE_LQR
                       else JointController())
                arm.attach_observer(observer, controller=ctl,
                                    encoder=Encoder())

            def abort_now():
                return pending() or not viewer.is_running()

            clear_table(model, data, arm, camera, cam_id, rng, reg,
                        viewer=viewer,
                        aborted=abort_now,
                        spawn_request=None,
                        estop=estop)

            arm.detach_observer()

            if not viewer.is_running():
                break

            # Decide what the interrupted search means, in priority order
            if take('reset'):
                hard_reset()
            elif take('place'):
                say('SPAWN  (6 during search) -- aborting, returning home')
                place_cube(model, data, reg, rng, viewer=viewer,
                           stop=lambda: cmd['reset'], estop=estop)
                say('cube teleported out of the gripper onto the table')
                enter_wait('after P')
            elif take('start'):
                say('STOP  (7) -- placing the cube back on the table')
                put_cube_down(model, data, arm, reg, rng, viewer=viewer,
                              estop=estop)
                enter_wait('after S stop')
            else:
                enter_wait('table clear')

        camera.renderer.close()


if __name__ == '__main__':
    main()
