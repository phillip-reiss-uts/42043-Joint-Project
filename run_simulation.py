#!/usr/bin/env python3
"""Simulation front-end: MuJoCo viewer, keyboard, scene, moving cube.

DESCRIPTION
-----------
The counterpart of run_hardware.py. It builds the same robot (`Arm`) and the
same task policy (`TaskStateMachine`) and the same pick sequence (`src/pick.py`),
and supplies the three providers for the simulator:

    simulation                       hardware
    ----------                       --------
    MujocoKinematics                 URKinematics   (chain FK + analytic M/g)
    MujocoBackend                    URBackend      (RTDE servoj + RG2)
    SimObjectWorld                   RealObjectWorld (RealSense depth)

Nothing above that layer changes, which is the point of the seams. This module
is the composition root: it chooses the providers, wires the callbacks, and
runs the viewer.

WHAT IS NOT PORTABLE, AND IS NOT HERE
-------------------------------------
THIS FILE IS SIMULATION-ONLY. It may import mujoco, the viewer, the object
pool and the cube driver. Nothing here is expected to run on hardware. The
task logic it drives -- WAIT/SEARCH, pause, abort -- lives in src/task.py and
is hardware-portable. The split is deliberate:

    run_simulation.py   viewer, keys, scene reset, cube motion   (sim only)
    src/task.py         state machine, pause semantics           (portable)
    src/arm.py          joint moves, gripper, grasp readback     (portable)
    src/servo.py        visual servo, occlusion recovery         (portable)
    src/world.py        the boundary between the two

RUNNING
-------
Run directly, or via `python main.py` / `python main.py --sim`:

    python run_simulation.py
"""

import numpy as np
import mujoco
import mujoco.viewer

from src.camera import RGBDCamera
# Explicit rather than `import *`, so this file's dependence on the scene and
# demonstration constants is legible at a glance.
from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, PARK_POS, POOL_SIZE, RAPID, SYNC_STEPS,
    USE_LQR, USE_OBSERVER, VERIFY_IK,
)
from src import config as cfg
from src.model_loader import load_robot_model
from src.camera import find_camera
from src.arm import Arm
# The simulation decides which providers the robot layer gets. Arm itself
# imports neither of these.
from src.kinematics_mujoco import MujocoKinematics
from src.robot_backend_mujoco import MujocoBackend
from src.pick_place import (ObjectRegistry, reset_scene, place_cube,
                            put_cube_down, clear_table)
from src.world_sim import SimObjectWorld
from src.overlay import Overlay
from src.task import Commands, TaskStateMachine, WAIT, HOLD
from src import target_motion
from src.target_motion import TargetMover


def _log(msg):
    # Optional diagnostics; gated by config.VERBOSE.
    if cfg.VERBOSE:
        print(msg)


def main(sensor_size=(320, 240), lightweight=False):
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

    # ---- COMMANDS --------------------------------------------------------
    # Keys 6-9 plus '.': the viewer maps a-z onto its 26 render flags (S toggles
    # shadows, E constraints, P contact-split, R reflection) and 0-5 onto geom
    # groups, so no letter or low digit is available.
    cmd = Commands()
    ui = {'overlay': None}

    def on_key(keycode):
        if keycode == ord('6'):
            cmd.place = True
        elif keycode == ord('7'):
            # Key 7 will PICKUP. The robot tracks whatever is on the table and
            # only grasps on this explicit command. Recovery searches happen
            # inside a pick.
            cmd.pickup = True
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

    backend = MujocoBackend(model, data, ARM_DOF)
    arm = Arm(model, data,
              kinematics=MujocoKinematics(model, backend.tool_body_id, ARM_DOF),
              backend=backend)

    if VERIFY_IK:
        arm.verify_ik()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        # Live overlays. Wraps viewer.sync, so every sync anywhere in the code
        # draws the overlay first -- no plumbing through the call chain.
        #
        # lightweight drops the two camera panels, each of which is its own GL
        # render per frame. The robot does not use them; 
        # Useful in resource constrained environments.
        overlay = Overlay(viewer, model, data, camera, cam_id, arm.base_id,
                          lightweight=lightweight)
        overlay.set_info('mode', 'WAIT', 'scene empty')
        name, speed = target_motion.current_band()
        overlay.set_info('move', 'obj motion %s' % name,
                         '%.0f mm/s' % (speed * 1e3))
        if lightweight:
            print('[INFO] lightweight: camera image and point-cloud panels are '
                  'off (two fewer GL renders per frame)')
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

        print('[INFO] 6 = spawn cube, 7 = pick up / put down (toggle), '
              '8 = pause, 9 = reset scene, . = object motion band')
        print('[STATE] WAIT  (scene empty; press 6 to spawn the cube. '
              'The robot then tracks it automatically; 7 grabs it and holds it '
              'at home until 7 is pressed again)')

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
            track_step=lambda: None,       # bound below
            pick_object=lambda: False,     # bound below
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
                    model, data, data.qpos[:ARM_DOF], dt=backend.dt,
                    lqr_weights=(lqr_weights() if USE_LQR else None))
                ctl = (LQRController(observer) if USE_LQR
                       else JointController())
                arm.attach_observer(observer, controller=ctl,
                                    encoder=Encoder())

            def abort_now():
                return cmd.pending() or not viewer.is_running()

            # One attempt: this is the recovery after a failed grip, not an
            # operator-driven "empty the table" loop. With no chute the table
            # never empties, so an unbounded loop would spin to MAX_PASSES.
            clear_table(model, data, arm, world, rng,
                        viewer=viewer, aborted=abort_now, estop=estop,
                        passes_max=1)
            arm.detach_observer()

        def put_down():
            # Return the cube to the xy it was picked from. The arm waits in
            # HOLD at the home pose, so using the current tool xy here would
            # place the cube at home instead.
            if last_pick_xy[0] is not None:
                xy = last_pick_xy[0]
            else:
                p_tool, _ = arm.kin.pose(data.qpos[:ARM_DOF])
                xy = (float(p_tool[0]), float(p_tool[1]))
            ok = put_cube_down(model, data, arm, reg, rng, viewer=viewer,
                               estop=estop, world=world, xy=xy,
                               stop=stopped)
            if ok:
                last_pick_xy[0] = None
            return ok

        # ---- TRACK: follow the object, never descend ---------------------
        #
        # A long-lived servo, not one per frame: the Kalman filter it holds is
        # the target's velocity estimate, and rebuilding it each frame would
        # throw that away and re-bootstrap from a single noisy look.
        from src.servo import PBVSServo
        from src.pick import pick_and_hold, search_for_object
        # Frame granularity for tracking.
        #
        # The chunk size does not change the result: servo() persists its
        # commanded joint vector across calls, so N iterations spread over any
        # number of calls gives the same motion as one call of N. Measured, with
        # the same 600 physics steps: peak-to-peak tool height was 0.62 mm at
        # chunk 1, 0.62 at 30, 0.69 at 100 and 0.63 at 300.
        #
        # This value is chosen purely for responsiveness: small enough that a
        # frame returns quickly, large enough that the viewer is not synced on
        # every physics step. 60 iterations is 0.12 s of sim per frame, and the
        # camera render (~17 ms) is matched to it so every frame has one render
        # rather than a burst.
        TRACK_CHUNK_ITERS = 60
        TRACK_SLICE_S = 2.0
        tracker = PBVSServo(
            model, data, arm, world, smooth=1.0,
            # One camera render per tracking chunk. The render is ~41 ms of a
            # ~55 ms chunk, so rendering every chunk keeps it steady; a cadence
            # shorter than the chunk would double the stalls, a longer one would
            # make some frames render and others not.
            perceive_every=TRACK_CHUNK_ITERS,
            v_max=cfg.SERVO_V_TRANSIT, v_descent=cfg.SERVO_V_DESCENT,
            occlude_ratio=cfg.SERVO_OCCLUDE_RATIO,
            occlude_recover=cfg.SERVO_OCCLUDE_RECOVER,
            occlude_fails=cfg.SERVO_OCCLUDE_FAILS,
            retreat_step=cfg.SERVO_RETREAT_STEP,
            retreat_max=cfg.SERVO_RETREAT_MAX,
            retreat_v=cfg.SERVO_RETREAT_V)

        # servo() blocks until it converges, which would freeze the viewer, so
        # tracking runs in chunks of TRACK_CHUNK_ITERS. The chunk size and the
        # wall-clock abort are both explained above; the abort exists only so a
        # genuine hang returns, and must stay far longer than one chunk (the
        # servo's entry work -- observe, filter setup -- alone can exceed a
        # quarter second, which is what made a 0.25 s cap fire before the first
        # step and leave the arm creeping toward a target it never reached).
        # Set DSH_TRACK_DEBUG=1 to print one line per tracking chunk.
        import os as _os
        TRACK_DEBUG = [_os.environ.get('DSH_TRACK_DEBUG', '') == '1']

        def _time_slice(seconds):
            import time as _time
            deadline = _time.perf_counter() + seconds
            return lambda: _time.perf_counter() >= deadline

        def track_step():
            def see():
                return [True] if tracker.observe() else []

            def reacquire():
                return search_for_object(model, data, arm, see, viewer=viewer,
                                         estop=estop,
                                         stop=lambda: not viewer.is_running(),
                                         wide=True)

            # HOVER_TOOL0_Z means "hover, do not descend": the same servo the
            # pick uses, at camera height rather than the grasp pose.
            _p0 = np.array(data.xpos[arm.base_id], float)
            tracker.servo(cfg.SIM_HOVER_TOOL0_Z,
                          max_iters=TRACK_CHUNK_ITERS,
                          viewer=viewer, estop=estop,
                          aborted=_time_slice(TRACK_SLICE_S),
                          v_max=cfg.SERVO_V_TRANSIT, track=True,
                          stationary_s=cfg.SERVO_STILL_TIME,
                          search=reacquire)
            if TRACK_DEBUG[0]:
                _p1 = np.array(data.xpos[arm.base_id], float)
                _R = np.array(data.xmat[arm.base_id], float).reshape(3, 3)
                _pd, _ = tracker.desired_pose(cfg.SIM_HOVER_TOOL0_Z)
                print('[TRACK] tool z %.4f -> %.4f (d %.4f)  axis=%s  '
                      'p_des=%s  err=%.4f' % (
                          _p0[2], _p1[2], _p1[2] - _p0[2],
                          np.round(_R[:, 2], 3), np.round(_pd, 3),
                          float(np.linalg.norm(_p1 - _pd))))

        # ---- PICK: one explicit grasp attempt ----------------------------
        # The put-down must return the cube to where it was picked. The arm is
        # carried to the home pose while holding, so the tool xy at PUTDOWN is
        # the home xy, not the pick xy.
        last_pick_xy = [None]

        def pick_object():
            active = world.active()
            if not active:
                return False
            i = active[0]
            last_pick_xy[0] = tuple(reg.pose(data, i)[:2])
            import time as _time
            # An explicit pickup must not turn into an indefinite chase. If the
            # object will not hold still, the servo would otherwise spend
            # SERVO_TRACK_ITERS (~2 min of sim time) waiting before giving up,
            # which reads as a hang. Bound the whole attempt by wall time and
            # let the caller treat it as a failed grip -- which re-homes and
            # searches, exactly the documented failed-grip path.
            #
            # THIS IS A HANG GUARD, NOT A PERFORMANCE TARGET, and it is WALL
            # time, so the machine decides how much pick fits inside it. A
            # successful pick is ~9.5 s of CPU headless, but every servo chunk
            # also syncs the viewer, draws the camera image and (OVERLAY_CLOUD)
            # does an extra depth pass for the point cloud. On the cell's
            # ThinkPad 120 s expired just after the jaws closed on a perfect
            # grasp -- [GRASP] held, then "pick exceeded its wall-clock budget"
            # -- and the pick was abandoned and re-homed every time.
            #
            # So it lives in config as PICK_BUDGET_S and is generous by default.
            # If a pick is abandoned, the message prints how long it took, which
            # separates "slow machine" from "genuine hang" without guesswork.
            # Lowering it is the wrong fix if picks are being cut off; the
            # overlay is the expensive part, not the control loop.
            t_pick = _time.perf_counter()
            mission_deadline = t_pick + cfg.PICK_BUDGET_S

            def give_up():
                # A repeat pickup key press must not abort the grasp that the
                # previous press started. Place/reset still pre-empt, and the
                # wall-clock budget still bounds a genuine hang.
                return (cmd.place or cmd.reset or not viewer.is_running()
                        or _time.perf_counter() >= mission_deadline)

            # Grasp, lift, and carry the cube to the home pose with the jaws
            # still shut. It is deliberately NOT released here: the pickup
            # command means "take it". The cube stays held at home until the
            # key is pressed again (HOLD -> PUTDOWN -> put_down below).
            ok = pick_and_hold(model, data, arm, world, i, home_q,
                               viewer=viewer, aborted=give_up, estop=estop,
                               hover_z=cfg.SIM_HOVER_TOOL0_Z)
            if not ok and _time.perf_counter() >= mission_deadline:
                print('[PICK] abandoned after %.1f s of a %.1f s budget -- this '
                      'is the HANG GUARD, not a failed grasp; raise '
                      'PICK_BUDGET_S in src/config.py if the grasp was held'
                      % (_time.perf_counter() - t_pick, cfg.PICK_BUDGET_S))
            return bool(ok)

        def on_reset():
            # Instant: the arm is teleported, not driven. Every parameter is
            # cleared, including the pause flag and the command queue.
            mujoco.mj_resetDataKeyframe(model, data, home_id)
            mujoco.mj_forward(model, data)
            # Drop any carry BEFORE the objects are parked. The hold loops step
            # through the backend, so a coupling left over from an aborted pick
            # would otherwise be re-applied and yank a parked cube to the tool.
            arm.detach_object()
            for i in range(len(reg.body_ids)):
                reg.teleport(data, i, PARK_POS)
                reg.state[i] = 'parked'
            mujoco.mj_forward(model, data)
            task.hard_reset()
            say('WAIT  (reset: cube parked, arm at home, all state cleared)')

        # Wire the scene-specific behaviour in, and clear the placeholders.
        task._go_home = go_home
        task._place_object = place_object
        task._track_step = track_step
        task._pick_object = pick_object
        task.on_reset = on_reset
        task.on_recover_search = run_search
        # Second press of the pickup key: put the held cube back on the table.
        task.on_put_object = put_down

        demo_idx = [None]
        # Latched arm pose for the WAIT hold, re-taken on each WAIT entry.
        # None means "capture on the next frame".
        wait_hold = [None]
        # Same, for HOLD: the pose the arm waits in with the cube in its jaws.
        held_hold = [None]

        while viewer.is_running():

            # The moving-target demo: decide WHICH object glides. The world
            # owns the mover and the physics loop steps it, so the robot is not
            # involved in the selection either.
            #
            # While the robot is HOLDING the cube there is nothing to glide: the
            # only active object is the one in the jaws, and stepping its mover
            # would drag it out of the gripper.
            act = world.active()
            if act and not task.holding:
                if demo_idx[0] != act[0]:
                    world.set_mover(TargetMover(model, data, reg, act[0],
                                    dt=backend.dt))
                    demo_idx[0] = act[0]
            else:
                world.set_mover(None)
                demo_idx[0] = None

            if task.mode == WAIT:
                # Nothing to follow: hold the home pose, keep the object demo
                # running so a spawn still glides.
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
                    # backend.step(), NOT a bare mj_step: the backend is what
                    # re-writes a carried object's pose each cycle, so a bare
                    # step silently drops the coupling and the cube falls out.
                    backend.step()
                viewer.sync()
            elif task.mode == HOLD:
                # Waiting with the cube in the jaws. The arm holds its latched
                # pose with the fingers shut; the physics still runs so the
                # viewer stays live and the grasp keeps being read back.
                #
                # No mover step here: the held cube must not be driven.
                if held_hold[0] is None:
                    held_hold[0] = arm.hold_pose()
                for _ in range(SYNC_STEPS):
                    arm.hold(GRIP_CLOSED, held_hold[0])
                    # backend.step() is the whole fix. A bare mj_step here left
                    # the carried cube as an ordinary free body for as long as
                    # HOLD lasted -- the state the operator is actually looking
                    # at -- so gravity walked it out of the jaws. Measured over
                    # 1000 steps of this loop: 2.74 mm of drift against the
                    # tool with mj_step, 0.12 mm with backend.step().
                    backend.step()
                viewer.sync()
            else:
                # TRACK and PICK drive the arm themselves (through the servo,
                # which steps the world and syncs the viewer). Leaving these
                # modes drops the latched poses so the next entry captures a
                # fresh one.
                wait_hold[0] = None
                held_hold[0] = None

            if not task.step():
                break

        camera.renderer.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
