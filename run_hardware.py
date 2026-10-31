#!/usr/bin/env python3
"""Hardware front-end: UR3e + OnRobot RG2 + D435i, no simulator.

DESCRIPTION
-----------
The counterpart of run_simulation.py. It builds the same robot (`Arm`) and the
same task policy (`TaskStateMachine`) and the same pick sequence (`src/pick.py`),
and swaps the three providers:

    simulation                       hardware
    ----------                       --------
    MujocoKinematics                 URKinematics   (chain FK + analytic M/g)
    MujocoBackend                    URBackend      (RTDE servoj + RG2)
    SimObjectWorld                   RealObjectWorld (RealSense depth)

Nothing above that layer changes, which is the point of the seams. This module
is the composition root: it chooses the providers, wires the callbacks, and
runs the operator console. It imports no MuJoCo.

WHAT IS NOT PORTABLE, AND IS NOT HERE
-------------------------------------
The object pool, spawning, the chunky tabletop and the moving-target demo live
in `pick_place.py` / `target_motion.py` / `run_simulation.py`. A real cell has a
table and a person putting a cube on it; there is nothing to spawn.

SAFETY
------
This drives a real arm. There is no e-stop button in this program: the UR
teach pendant's own stop is the authority. Ctrl-C  QUITS the loop 
it still cannot stop a `servoj` stream the instant it is sent, so press the pendant stop 
if the arm is moving. Speed is deliberately slow (FEED); raise it only after the geometry checks pass.
"""

import argparse
import os
import sys
import time

import numpy as np

from src import config as cfg
from src.arm import Arm
from src.arm_dynamics import ArmDynamics, rot_from_quat
from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, RAPID, FEED,
    TABLE_HEIGHT, OBJECT_HEIGHT, TIP_OFFSET, APPROACH_Z, STEPS_RELEASE,
)
from src.kinematics_ur import URKinematics
from src.pick import pick_and_hold
from src.task import Commands, TaskStateMachine, TRACK, WAIT
from src.world_real import (FixedCameraWorld, RealObjectWorld, RealSenseSource,
                            camera_transform)


class VisibleWorld:
    # RealObjectWorld plus the one scene question the task policy asks:
    # "is there anything on the table?".
    #
    # The boundary deliberately has no `active()` -- a hardware world cannot
    # enumerate an object pool -- so this caches the last camera look. "Active"
    # therefore means "the camera saw something the last time we asked", which
    # is the honest hardware equivalent.

    def __init__(self, world, cache_frames=False):
        self.world = world
        self.last = []
        # The snapshot feature needs the frame that produced the last views. It
        # stays OFF unless run_hardware was given --snapshot, so a normal run
        # keeps nothing.
        if hasattr(world, 'cache_frames'):
            world.cache_frames = bool(cache_frames)

    def last_perception(self):
        # Forwarded, because this wrapper is the object the front-end holds.
        getter = getattr(self.world, 'last_perception', None)
        return None if getter is None else getter()

    def refresh(self):
        self.last = self.world.perceive()
        return self.last

    # ---- ObjectWorld ----
    def perceive(self):
        self.last = self.world.perceive()
        return self.last

    def gripper_pose(self):
        return self.world.gripper_pose()

    # ---- scene questions (hardware meaning) ----
    def active(self):
        return [0] if self.last else []

    def close(self):
        src = getattr(self.world, 'src', None)
        if src is not None:
            src.close()


def _poll_key():
    # Non-blocking single keypress. msvcrt on Windows, select on POSIX.
    try:
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None
    except ImportError:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    return None


def _camera_mount():
    # gripper base -> camera optical frame, from the configured bracket.
    T = np.eye(4)
    T[:3, :3] = rot_from_quat(cfg.CAM_MOUNT_QUAT)
    T[:3, 3] = cfg.CAM_MOUNT_POS
    return T


def _fixed_camera_transform(): # DEPRECATED
    # camera -> base, from a measured position and aim point.
    pos = np.array(cfg.FIXED_CAM_POS, float)
    target = np.array(cfg.FIXED_CAM_TARGET, float)
    return camera_transform(pos, target - pos, cfg.FIXED_CAM_UP)


def _put_down(arm):
    # Lower the held cube to the table directly beneath the gripper and release.
    #
    # The release height is TIP_OFFSET above the cube's resting centre, which is
    # the same relation the grasp used -- see put_cube_down in pick_place for the
    # simulation twin (that one also clamps into the spawn region, which has no
    # meaning here).
    p, R = arm.kin.pose(arm.backend.joint_positions())
    release_z = TABLE_HEIGHT + OBJECT_HEIGHT / 2.0 + TIP_OFFSET
    q_high = arm.solve_ik([p[0], p[1], release_z + APPROACH_Z],
                          arm.backend.joint_positions(), R_target=R)
    arm.move(q_high, GRIP_CLOSED, rate=RAPID)
    q_low = arm.solve_ik([p[0], p[1], release_z], q_high, R_target=R)
    arm.move(q_low, GRIP_CLOSED, rate=FEED)

    def settle(q, seconds=0.30):
        # move() returns when the last target is SENT; the arm is still moving.
        # Opening there releases the cube in mid-air, which is the "drop".
        for _ in range(max(1, int(seconds / arm.backend.dt))):
            arm.backend.command(q, GRIP_CLOSED)
            arm.backend.step()

    settle(q_low)
    # The arm settles ~11 mm ABOVE the pose it is given, so correct the measured
    # residual before letting go: command the release height that far lower.
    p_now, _ = arm.kin.pose(arm.backend.joint_positions())
    dz = float(p_now[2]) - release_z
    if 0.002 < dz < 0.030:
        print('[HW] put-down settled %.0f mm high -- correcting' % (dz * 1e3))
        q_low = arm.solve_ik([p[0], p[1], release_z - dz],
                             arm.backend.joint_positions(), R_target=R)
        arm.move(q_low, GRIP_CLOSED, rate=FEED)
        settle(q_low)

    arm.open_gripper(STEPS_RELEASE)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog='run_hardware.py',
                                 description='Run the pick on a real UR3e.')
    ap.add_argument('--ip', default=cfg.UR_IP, help='controller IP')
    ap.add_argument('--no-gripper', action='store_true',
                    help='arm only; grasp checks report "unknown"')
    ap.add_argument('--snapshot', metavar='DIR', default=None,
                    help='enable the v key: write an annotated depth PNG of the '
                         'last camera look into DIR. OFF by default -- no frame '
                         'is cached and src/liveview is never imported unless '
                         'this is given')
    args = ap.parse_args(argv)

    print()
    print('=' * 74)
    print('HARDWARE  UR3e + OnRobot RG2 + D435i')
    print('=' * 74)
    print('  controller  : %s' % args.ip)
    print('  cycle       : %.1f Hz (servoj period %.3f s)'
          % (1.0 / cfg.UR_CYCLE_DT, cfg.UR_CYCLE_DT))
    print('  the pendant stop is the only e-stop; speed starts at FEED')
    # Say the perception floor out loud: it is the setting most likely to make
    # the cube vanish from the log, and it is invisible otherwise.
    print('  perception : hand filter %.0f mm (points below table+%.0f mm '
          'ignored), log every %.1f s'
          % (cfg.HAND_FILTER_MM, cfg.HAND_FILTER_MM, cfg.PERCEPTION_LOG_S))
    print('  snapshots  : %s'
          % ('v writes a PNG into %s' % args.snapshot if args.snapshot
             else 'OFF (add --snapshot DIR to arm the v key)'))
    print()

    # ---- providers -------------------------------------------------------
    try:
        backend = _connect(args)
    except Exception as exc:
        print('[HW] cannot connect: %s' % exc)
        print('     install ur_rtde and check the IP, then retry')
        return 2

    dyn = ArmDynamics.from_file(_links_path())
    kin = URKinematics(dyn, (np.full(ARM_DOF, cfg.UR_SERVO_KP),
                             np.full(ARM_DOF, cfg.UR_SERVO_KV)),
                       tool_pos=cfg.TOOL_MOUNT_POS,
                       tool_quat=cfg.TOOL_MOUNT_QUAT,
                       root_yaw=cfg.UR_ROOT_YAW)
    arm = Arm(None, None, kinematics=kin, backend=backend)

    try:
        depth = RealSenseSource()
    except Exception as exc:
        print('[HW] RealSense unavailable: %s' % exc)
        print('     install pyrealsense2 (and librealsense), then retry')
        backend.close()
        return 2

    flange_pose = lambda: kin.pose(backend.joint_positions())
    # cache_frames is the snapshot switch: True only when --snapshot was given.
    if cfg.CAMERA_MODE == 'fixed':
        T_base_cam = _fixed_camera_transform()
        world = VisibleWorld(FixedCameraWorld(depth, flange_pose=flange_pose,
                                              T_base_cam=T_base_cam),
                             cache_frames=bool(args.snapshot))
        print('[HW] camera: fixed (eye-to-hand) at %s, aimed at %s'
              % (np.round(cfg.FIXED_CAM_POS, 3),
                 np.round(cfg.FIXED_CAM_TARGET, 3)))
    else:
        world = VisibleWorld(RealObjectWorld(depth, flange_pose=flange_pose,
                                             T_flange_cam=_camera_mount()),
                             cache_frames=bool(args.snapshot))
        print('[HW] camera: wrist-mounted (eye-in-hand)')

    # Snapshot writer only. A normal run
    # never imports src.liveview and never touches cv2.
    shots = None
    if args.snapshot:
        from src.liveview import Snapshotter
        shots = Snapshotter(args.snapshot)

    # ---- task ------------------------------------------------------------
    cmd = Commands()

    print('[HW] moving to the home configuration')
    arm.move(cfg.UR_HOME_Q, GRIP_OPEN, rate=RAPID)

    def say(msg):
        print('[STATE] %s' % msg)

    def at_home():
        return float(np.linalg.norm(backend.joint_positions()
                                    - np.asarray(cfg.UR_HOME_Q))) < 0.02

    def go_home():
        arm.move(cfg.UR_HOME_Q, GRIP_OPEN, rate=RAPID)

    def place_object():
        # Nothing is spawned on hardware: the operator puts the cube down.
        where = 'fixed camera' if cfg.CAMERA_MODE == 'fixed' else 'wrist camera'
        print('[HW] place the cube in view of the %s' % where)

    def track_step():
        # One short closed-loop look-and-follow chunk. The task calls this from
        # TRACK; on hardware the operator-paced loop means one chunk is enough.
        from src.servo import PBVSServo
        sv = PBVSServo(None, None, arm, world)
        sv.servo(cfg.SERVO_STANDOFF, max_iters=cfg.UR_TRACK_CHUNK)

    def pick_object():
        if not world.refresh():
            return False
        return pick_and_hold(None, None, arm, world, 0, cfg.UR_HOME_Q)

    task = TaskStateMachine(
        commands=cmd, say=say, at_home=at_home, go_home=go_home,
        place_object=place_object, track_step=track_step,
        pick_object=pick_object, running=lambda: True,
        arm=arm, world=world)
    task.on_put_object = lambda: _put_down(arm)

    # START IN WAIT, AND STAY THERE. The arm holds the home pose until an
    # explicit command, but the camera still looks once a second (see the loop
    # below), so a cube placed in view is reported the moment it appears. It is
    # NOT tracked until `t` releases the hold; nothing perceives while the task
    # is in WAIT, so the entering of TRACK can only be done on that command.
    sighted = [False]

    # ---- operator console ------------------------------------------------
    print()
    print('  p = pick up / put down (toggle)')
    print('  h = go home    s = status    q or Ctrl-C = quit')
    print('  space = pause / resume (holds position; p is ignored while paused)')
    print('  t = start / stop tracking (starts held; nothing moves until t)')
    if shots is not None:
        print('  v = save an annotated depth PNG of the last camera look')
    print()
    print('[STATE] WAIT  (place the cube in view; it is reported, then press t '
          'to track it, or p to pick it)')

    held_q = np.array(backend.joint_positions(), float)
    grip = GRIP_OPEN
    mode_seen = None
    paused = False
    track_on = False
    last_look = 0.0

    while True:
        # Keep the servoj stream alive: a real controller expects a command
        # every cycle, unlike the simulator where stepping is an internal call.
        backend.command(held_q, grip)
        backend.step()

        key = _poll_key()
        if key in ('q', 'Q', '\x1b'):
            break
        if key in ('h', 'H'):
            arm.move(cfg.UR_HOME_Q, GRIP_OPEN, rate=RAPID)
            held_q = np.array(backend.joint_positions(), float)
            grip = GRIP_OPEN
        elif key in ('p', 'P'):
            if paused:
                # Otherwise the command would fire the moment we resume, with
                # your hand possibly still in the cell.
                print('[HW] paused -- press space first, then p')
            else:
                world.refresh()
                cmd.pickup = True
        elif key in ('s', 'S'):
            p, _ = kin.pose(backend.joint_positions())
            print('[STATE] %s  holding=%s  paused=%s  tool=(%.3f, %.3f, %.3f)'
                  % (task.mode, task.holding, paused, p[0], p[1], p[2]))
        elif key in ('v', 'V'):
            # Snapshot of the LAST camera look. Deliberately not a fresh capture:
            # this must not consume a frame perception wanted, and it must never
            # wait on the camera inside the operator loop.
            if shots is None:
                print('[HW] snapshots are off -- restart with --snapshot DIR')
            else:
                frame = world.last_perception()
                if frame is None:
                    print('[HW] no camera look yet -- press p first')
                else:
                    from src.liveview import annotate
                    img = annotate(frame['z16'], frame['K'], frame['p_cam_w'],
                                   frame['R_wc'], frame['views'],
                                   note='%s  dets %d  holding=%s'
                                        % (task.mode, len(frame['views']),
                                           task.holding),
                                   tool_pose=kin.pose(
                                       backend.joint_positions())[0])
                    print('[HW] wrote %s' % shots.save(img))
        elif key in ('\r', '\n'):
            pass
        elif key == ' ':
            # PAUSE / RESUME. The arm keeps its last commanded target (the
            # servoj stream above is what holds it there), the mode stops being
            # advanced, and no pick can start. It does NOT abort a pick that is
            # already running: task.step() blocks for the whole pick and the key
            # is only read between them, so the pendant is still the way out of
            # a pick in progress.
            paused = not paused
            held_q = np.array(backend.joint_positions(), float)
            grip = GRIP_CLOSED if task.holding else GRIP_OPEN
            print('[HW] %s -- %s'
                  % ('PAUSED' if paused else 'RUNNING',
                     'holding position; p is ignored until you resume'
                     if paused else 'policy advancing'))
        elif key in ('t', 'T'):
            # TRACK / HOLD. Off means the policy stops advancing, so the arm
            # stays where it is instead of following the cube; on picks the
            # follow up again from there. It does NOT re-home and it does not
            # consume a look, so switching it back on resumes around the same
            # estimate.
            #
            # On is the explicit command to TRACK, and the only way the task
            # leaves WAIT on someone else's cue: releasing the hold while a cube
            # is in view enters TRACK here, rather than the task doing it the
            # moment it first sees one.
            track_on = not track_on
            held_q = np.array(backend.joint_positions(), float)
            print('[HW] tracking %s' % ('on (entering TRACK if a cube is in view)'
                                        if track_on else 'off -- holding'))
            if track_on and not paused and task.mode == WAIT:
                if world.refresh():
                    task.enter_track('object in view, tracking released')
                else:
                    print('[HW] no cube in view -- place one to track')

        # Advance the policy. This is where a pick runs; it blocks until the
        # grasp (or the put-down) finishes, streaming servoj targets itself.
        # Holding it while tracking is off lets the front-end own the
        # WAIT -> TRACK transition. In WAIT, look once a second: the hardware
        # world has no object pool, so active() only changes when something
        # looks, and the task never looks for itself in WAIT. Each look costs
        # ~65 ms, which is fine while holding still.
        #   tracking off -> report a cube that appears or leaves, stay in WAIT
        #   tracking on  -> a cube in view is enough to start tracking
        if track_on and not paused and task.mode == WAIT:
            now = time.monotonic()
            if now - last_look >= 1.0:
                last_look = now
                seen = bool(world.perceive())
                if seen != sighted[0]:
                    sighted[0] = seen
                    print('[HW] cube %s' % ('in view -- p picks it up, t starts '
                                            'tracking it'
                                            if seen else 'no longer in view'))
        if paused or not track_on:
            # Still streaming held_q at the top of the loop; just do not let the
            # policy run. "continue" is what keeps the servoj stream fed.
            continue

        try:
            task.step()
        except KeyboardInterrupt:
            # QUIT, do not swallow it. This used to `continue`, which made
            # Ctrl-C during a pick look like nothing happened: the pick was
            # abandoned mid-motion and the loop simply started the next one.
            print('\n[HW] interrupted -- stopping')
            break

        # Re-latch the hold pose on every mode change, so the stream does not
        # turn into an integrator under the arm's own compliance.
        if task.mode != mode_seen:
            mode_seen = task.mode
            held_q = np.array(backend.joint_positions(), float)
            grip = GRIP_CLOSED if task.holding else GRIP_OPEN

    print('[HW] closing')
    backend.close()
    world.close()
    return 0


def _links_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'models', 'ur3e', 'arm_dynamics.json')


def _connect(args):
    from src.robot_backend_ur import connect
    return connect(args.ip, activate_gripper=not args.no_gripper,
                   dt=cfg.UR_CYCLE_DT,
                   lookahead_time=cfg.UR_LOOKAHEAD_TIME,
                   gain=cfg.UR_SERVO_GAIN)


if __name__ == '__main__':
    raise SystemExit(main())
