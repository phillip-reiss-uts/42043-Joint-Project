# Simulation entry point: MuJoCo viewer, keyboard, scene, moving cube.
#
# THIS FILE IS SIMULATION-ONLY. It may import mujoco, the viewer, the object
# pool and the cube driver. Nothing here is expected to run on hardware.
#
# The task logic it drives -- WAIT/SEARCH, pause, abort -- lives in
# src/task.py and is hardware-portable. The split is deliberate:
#
#   run_simulation.py   viewer, keys, scene reset, cube motion   (sim only)
#   src/task.py         state machine, pause semantics           (portable)
#   src/arm.py          joint moves, gripper, grasp readback     (portable)
#   src/servo.py        visual servo, occlusion recovery         (portable)
#   src/world.py        the boundary between the two
#
# Run directly, or via `python main.py` / `python main.py --sim`:
#
#     python run_simulation.py

import numpy as np
import mujoco
import mujoco.viewer

from src.camera import RGBDCamera
# Explicit rather than `import *`, so this file's dependence on the scene and
# demonstration constants is legible at a glance.
from src.config import (
    ARM_DOF, GRIP_OPEN, PARK_POS, POOL_SIZE, RAPID, SYNC_STEPS,
    USE_LQR, USE_OBSERVER, VERIFY_IK,
)
from src import config as cfg
from src.model_loader import load_robot_model
from src.perception import find_camera
from src.arm import Arm
from src.pick_place import (ObjectRegistry, reset_scene, place_cube,
                            put_cube_down, clear_table)
from src.world import SimObjectWorld
from src.overlay import Overlay
from src.task import Commands, TaskStateMachine, WAIT
from src import target_motion
from src.target_motion import TargetMover


def _log(msg):
    # Optional diagnostics; gated by config.VERBOSE.
    if cfg.VERBOSE:
        print(msg)


def main(sensor_size=(320, 240)):
    model = load_robot_model()
    if model is None:
        return 1

    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_id < 0:
        print('[ERROR] model has no "home" keyframe; cannot start.')
        return 1
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    home_st = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home_st, home_id)
    home_q = np.array(home_st.qpos[:ARM_DOF])

    cam_name, cam_id = find_camera(model)
    if cam_id < 0:
        print('[ERROR] no wrist camera found; cannot proceed.')
        return 1
    _log(f'[INFO] {model.nbody} bodies, {model.njnt} joints, '
         f'{model.nu} actuators; wrist camera = {cam_name}')

    rng = np.random.default_rng()

    # 320x240 sensor: ~2x the position error of 640x480 (1.5 mm vs 0.75 mm
    # mean, still far inside grasp tolerance) but ~5x cheaper per frame, which
    # keeps the perception stall inside the servo loop below the visible range.
    camera = RGBDCamera(model, cam_name, width=sensor_size[0],
                        height=sensor_size[1])
    _log(f'[INFO] intrinsics K:\n{camera.K}')

    # COMMANDS
    # Keys 6-9 plus '.': the viewer maps a-z onto its 26 render flags (S toggles
    # shadows, E constraints, P contact-split, R reflection) and 0-5 onto geom
    # groups, so no letter or low digit is available to us.
    cmd = Commands()
    ui = {'overlay': None}

    def on_key(keycode):
        if keycode == ord('6'):
            cmd.place = True
        elif keycode == ord('7'):
            cmd.start = True
        elif keycode == ord('8'):
            cmd.pause = True
        elif keycode == ord('9'):
            cmd.reset = True
        elif keycode == ord('.'):
            # cycle the object's motion band: off -> slow -> medium -> fast
            name, speed = target_motion.cycle_band()
            print('[MOVE] object motion band: %-6s (%.0f mm/s)'
                  % (name, speed * 1e3))
            ov = ui['overlay']
            if ov is not None:
                ov.set_info('move', 'obj motion %s' % name,
                            '%.0f mm/s' % (speed * 1e3))

    arm = Arm(model, data)

    if VERIFY_IK:
        arm.verify_ik()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        # Live overlays. Wraps viewer.sync, so every sync anywhere in the code
        # draws the overlay first -- no plumbing through the call chain.
        overlay = Overlay(viewer, model, data, camera, cam_id, arm.base_id)
        overlay.set_info('mode', 'WAIT', 'scene empty')
        name, speed = target_motion.current_band()
        overlay.set_info('move', 'obj motion %s' % name,
                         '%.0f mm/s' % (speed * 1e3))
        ui['overlay'] = overlay
        # Arm pushes diagnostics through whatever has .overlay; it does not
        # import the overlay layer itself.
        arm.viewer = viewer

        reg = ObjectRegistry(model, POOL_SIZE)
        reset_scene(model, data, reg, rng, viewer=None)

        # The world: the robot's only view of what it is acting on. The object
        # pool, the camera and (for the demonstration) the cube's motion all
        # live behind this. The mover is owned here, not by the arm, and is
        # stepped from the physics loops.
        world = SimObjectWorld(model, data, camera, cam_id, reg)

        print('[INFO] 6 = spawn cube, 7 = start/stop search, 8 = pause, '
              '9 = reset scene, . = object motion band')
        print('[STATE] WAIT  (scene empty; press 6 to spawn the cube, '
              '7 to search)')

        def say(msg):
            print('[STATE] %s' % msg)
            overlay.set_info('mode', msg)

        def at_home():
            return float(np.linalg.norm(data.qpos[:ARM_DOF] - home_q)) < 1e-2

        # The task state machine is built FIRST, so the closures below can bind
        # their motion-loop callbacks to it (task.estop / task.stopped) instead
        # of referencing a `task` variable that does not exist yet. Referencing
        # it inside a closure happens to work by late binding, but it is
        # fragile and reads as a bug.
        task = TaskStateMachine(
            commands=cmd, say=say, at_home=at_home,
            go_home=lambda: None,          # bound below
            place_object=lambda: None,     # bound below
            clear_table=lambda: None,      # bound below
            running=lambda: viewer.is_running(),
            arm=arm, world=world)

        # Bound to the task now that it exists.
        estop, stopped = task.estop, task.stopped

        def go_home():
            arm.move(home_q, GRIP_OPEN, rate=RAPID, viewer=viewer,
                     estop=estop, stop=stopped)

        def place_object():
            idx = place_cube(model, data, reg, rng, viewer=viewer,
                             stop=lambda: cmd.reset, estop=estop,
                             world=world)
            overlay.set_info('task', 'spawned obj %s' % idx, 'on the table')
            say('WAIT  (cube placed on the table, index %s)' % idx)

        def run_search():
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
                return cmd.pending() or not viewer.is_running()

            clear_table(model, data, arm, world, rng,
                        viewer=viewer, aborted=abort_now, estop=estop)
            arm.detach_observer()

        def put_down():
            put_cube_down(model, data, arm, reg, rng, viewer=viewer,
                          estop=estop, world=world)

        def on_reset():
            # Instant: the arm is teleported, not driven. Every parameter is
            # cleared, including the pause flag and the command queue.
            mujoco.mj_resetDataKeyframe(model, data, home_id)
            mujoco.mj_forward(model, data)
            for i in range(len(reg.body_ids)):
                reg.teleport(data, i, PARK_POS)
                reg.state[i] = 'parked'
            mujoco.mj_forward(model, data)
            task.hard_reset()
            say('WAIT  (reset: cube parked, arm at home, all state cleared)')

        # Wire the scene-specific behaviour in, and clear the placeholders.
        task._go_home = go_home
        task._place_object = place_object
        task._clear_table = run_search
        task.on_reset = on_reset
        task.on_put_down = put_down

        demo_idx = [None]
        # Latched arm pose for the WAIT hold, re-taken on each WAIT entry.
        # None means "capture on the next frame".
        wait_hold = [None]

        while viewer.is_running():

            # The moving-target demo: decide WHICH object glides. The world
            # owns the mover and the physics loop steps it, so the robot is not
            # involved in the selection either.
            if task.mode == WAIT:
                act = world.active()
                if act:
                    if demo_idx[0] != act[0]:
                        world.set_mover(TargetMover(model, data, reg, act[0]))
                        demo_idx[0] = act[0]
                else:
                    world.set_mover(None)
                    demo_idx[0] = None

                # Step the same number of times per render as the motion loops
                # do. At one step per frame the sim runs ~0.2x real time, so the
                # object crawls here and appears to lurch the moment a search
                # starts.
                #
                # The hold target is latched per WAIT entry, NOT re-read each
                # step: arm.hold(GRIP_OPEN) with no target re-commands wherever
                # the arm already is, so any gravity sag becomes the new target
                # and the arm walks away. Measured: 148 mm of tool drift over a
                # WAIT period with the feedback form, ~12 mm with a latched one.
                if wait_hold[0] is None:
                    wait_hold[0] = arm.hold_pose()
                for _ in range(SYNC_STEPS):
                    world.step_mover()
                    arm.hold(GRIP_OPEN, wait_hold[0])
                    mujoco.mj_step(model, data)
                viewer.sync()
            else:
                # Leaving WAIT: drop the latched pose so the next WAIT entry
                # captures a fresh one (the arm will have moved by then).
                wait_hold[0] = None

            if not task.step():
                break

        camera.renderer.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
