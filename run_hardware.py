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
    PICK_Z, APPROACH_Z, STEPS_RELEASE,
)
from src.kinematics_ur import URKinematics
from src.pick import pick_and_hold
from src.servo import PBVSServo
from src.task import Commands, TaskStateMachine, TRACK, WAIT
from src.world_real import (RealObjectWorld, RealSenseSource)


ACTIVE_EMPTY_RUN = 3


class VisibleWorld:
    # RealObjectWorld plus the scene question the task asks: is anything on the
    # table? Derived from the last camera look, since a hardware world cannot
    # enumerate an object pool.

    def __init__(self, world, cache_frames=False):
        self.world = world
        self.last = []
        self._empty_run = 0
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
    def visible_now(self):
        # Whether the last look saw anything.
        return [0] if self.last else []

    def active(self):
        # Whether there is something on the table. Tolerates a run of empty
        # looks, so a cube leaving the frame does not empty the table.
        self._empty_run = 0 if self.last else self._empty_run + 1
        return [0] if self._empty_run < ACTIVE_EMPTY_RUN else []

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


def _put_down(arm, xy=None):
    # Lower the held cube to the table at the xy it was picked from and release.
    #
    # The descent matches the GRASP: PICK_Z, applied
    # in the FLANGE frame like every other height in config.
    #
    # See put_cube_down in pick_place for the simulation twin (that one also
    # clamps into the spawn region, which has no meaning here).
    p, R = arm.kin.pose(arm.backend.joint_positions())
    if xy is None:
        tx, ty = p[0], p[1]
    else:
        tx, ty = float(xy[0]), float(xy[1])
    release_z = PICK_Z
    q_high = arm.solve_ik_tcp([tx, ty, release_z + APPROACH_Z],
                              arm.backend.joint_positions(), R_target=R)
    arm.move(q_high, GRIP_CLOSED, rate=RAPID)
    q_low = arm.solve_ik_tcp([tx, ty, release_z], q_high, R_target=R)
    arm.move(q_low, GRIP_CLOSED, rate=FEED)

    def settle(q, seconds=0.30):
        # move() returns when the last target is SENT; the arm is still moving.
        # Opening there releases the cube in mid-air, which is the "drop".
        for _ in range(max(1, int(seconds / arm.backend.dt))):
            arm.command_step(q, GRIP_CLOSED)

    settle(q_low)
    # Report the measured release height rather than correcting it. The grasp
    # closes on the estimate the same way and is the reference: whatever
    # residual THIS cell leaves, the put-down is now at the grasp's height.
    p_now, _ = arm.kin.pose(arm.backend.joint_positions())
    miss = float(np.linalg.norm(
        np.asarray(p_now, float) - np.array([tx, ty, release_z], float)))
    print('[HW] put-down at %+.1f mm from the grasp height, %.1f mm from target'
          % ((float(p_now[2]) - release_z) * 1e3, miss * 1e3))
    if miss > 0.010:
        print('[HW] release NOT reached -- keeping the cube in the jaws')
        return False

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
    ap.add_argument('--servo-path', choices=('state-space', 'raw'),
                    default='raw',
                    help='raw: the original direct command path (default). '
                         'state-space: ArmObserver + LQR around the PBVS joint '
                         'reference.')
    ap.add_argument('--state-space', action='store_true',
                    help='alias for --servo-path state-space')
    ap.add_argument('--raw-servo', action='store_true',
                    help='alias for --servo-path raw (the default)')
    args = ap.parse_args(argv)
    if args.state_space:
        args.servo_path = 'state-space'
    if args.raw_servo:
        args.servo_path = 'raw'

    print()
    print('=' * 74)
    print('HARDWARE  UR3e + OnRobot RG2 + D435i')
    print('=' * 74)
    print('  controller  : %s' % args.ip)
    print('  cycle       : %.1f Hz (servoj period %.3f s)'
          % (1.0 / cfg.UR_CYCLE_DT, cfg.UR_CYCLE_DT))
    print('  hover       : tool0 %.3f m  camera %.3f m above table'
          % (cfg.HOVER_TOOL0_Z, cfg.CAM_HOVER_HEIGHT))
    print('  speed scale : %.3f' % cfg.UR_SPEED_SCALE)
    print('  servo path  : %s' % args.servo_path)
    print('  the pendant stop is the only e-stop; speed starts at FEED')
    # Say the perception floor out loud: it is the setting most likely to make
    # the cube vanish from the log, and it is invisible otherwise.
    print('  perception : hand filter %.0f mm (points below table+%.0f mm '
          'ignored), log every %.1f s'
          % (cfg.HAND_FILTER * 1e3, cfg.HAND_FILTER * 1e3, cfg.PERCEPTION_LOG_S))
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
    # FLANGE frame: everything measured here (CAM_MOUNT, PICK_Z) is stated from
    # the flange, which is what the controller reports. Servo targets and the
    # contact gate step back to gripper/base themselves.
    kin = URKinematics(dyn, (np.full(ARM_DOF, cfg.UR_SERVO_KP),
                             np.full(ARM_DOF, cfg.UR_SERVO_KV)),
                       tool_pos=(0.0, 0.0, 0.0),
                       tool_quat=(1.0, 0.0, 0.0, 0.0),
                       root_yaw=cfg.UR_ROOT_YAW)
    arm = Arm(None, None, kinematics=kin, backend=backend)

    if args.servo_path == 'state-space':
        print()
        print('[HW] STATE-SPACE PATH -- risks:')
        print('     - UR_SERVO_KP/KV are placeholders, not identified gains')
        print('     - the observer/LQR loop is not hardware-validated')
        print('     - it can fight the UR servo or oscillate at speed')
        print('     - run tools/validate_state_space.py fit AND ab first')
        answer = input("type 'state-space' to continue > ").strip().lower()
        if answer != 'state-space':
            print('[HW] refused; use the default raw path')
            backend.close()
            return 2
        from src.observer import ArmObserver
        from src.controller import (Encoder, JointController, LQRController,
                                    lqr_weights)
        q_now = np.asarray(backend.joint_positions(), float)
        observer = ArmObserver(
            None, None, q_now, dt=cfg.UR_CYCLE_DT, kin=kin,
            lqr_weights=(lqr_weights() if cfg.USE_LQR else None))
        ctl = LQRController(observer) if cfg.USE_LQR else JointController()
        arm.attach_observer(observer, controller=ctl,
                            encoder=Encoder(noisy=False))
        print('[HW] state-space: observer attached; UR_SERVO_KP/KV are '
              'placeholders -- validate at low speed')
    else:
        print('[HW] raw servo path: no observer/LQR')

    try:
        depth = RealSenseSource()
    except Exception as exc:
        print('[HW] RealSense unavailable: %s' % exc)
        print('     install pyrealsense2 (and librealsense), then retry')
        backend.close()
        return 2

    flange_pose = lambda: kin.pose(backend.joint_positions())
    # cache_frames is the snapshot switch: True only when --snapshot was given.
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
        # The tracker is created below; this closure only runs after that.
        # Its last command belongs to the pose we just left.
        tracker.reset_command()

    def place_object():
        # Nothing is spawned on hardware: the operator puts the cube down.
        print('[HW] place the cube in view of the wrist camera')

    # The pose streamed while the policy is idle. Defined before the closures
    # below because _relatch() rebinds it.
    held_q = np.array(backend.joint_positions(), float)
    grip = GRIP_OPEN

    def _relatch():
        # Re-read the hold pose after a servo chunk, which commands its own
        # targets.
        nonlocal held_q
        held_q = np.array(backend.joint_positions(), float)

    # ONE tracker for the whole session, not one per chunk. A new servo resets
    # _q_cmd, the KF, yaw_est and best_span every UR_TRACK_CHUNK, which makes the
    # arm re-seed its command from wherever it happens to be and loses the
    # velocity/yaw estimates the tracking loop depends on.
    #
    # The caps and cadence are explicit here because PBVSServo's defaults are
    # the simulator's: v_max=inf and perceive_every=15. The sim tracker passes
    # these values, so the hardware path must too or the two do different laws.
    tracker = PBVSServo(
        None, None, arm, world,
        perceive_every=cfg.SERVO_PERCEIVE_EVERY,
        v_max=cfg.SERVO_V_TRANSIT,
        v_descent=cfg.SERVO_V_DESCENT,
        occlude_ratio=cfg.SERVO_OCCLUDE_RATIO,
        occlude_recover=cfg.SERVO_OCCLUDE_RECOVER,
        occlude_fails=cfg.SERVO_OCCLUDE_FAILS,
        retreat_step=cfg.SERVO_RETREAT_STEP,
        retreat_max=cfg.SERVO_RETREAT_MAX,
        retreat_v=cfg.SERVO_RETREAT_V)

    def _reacquire():
        # One look for a lost target. observe() folds it into p_est/KF/yaw;
        # world.refresh() alone only updates the UI cache and leaves the servo
        # chasing its previous estimate.
        return bool(tracker.observe())

    def track_step():
        # One closed-loop look-and-follow chunk, called from TRACK.
        #
        # track and search are both required: the servo's lost-target branch is
        # gated on them.
        tracker.servo(cfg.HOVER_TOOL0_Z, max_iters=cfg.UR_TRACK_CHUNK,
                      track=True, search=_reacquire)
        if tracker.stage_obs == 0:
            print('[TRACK] chunk ended with no fresh look -- the cube is out of '
                  'view; the arm is held where it stopped')
        _relatch()

    # The xy the cube was picked from. The pick carries it to home, so this is
    # the only record of where it should be put back down.
    last_pick_xy = [None]

    def pick_object():
        views = world.refresh()
        if not views:
            return False
        pick_xy = np.array(views[0].centre[:2], float)
        ok = pick_and_hold(None, None, arm, world, 0, cfg.UR_HOME_Q)
        # The pick moved the arm to home; do not resume tracking from the
        # pre-pick command vector.
        tracker.reset_command()
        last_pick_xy[0] = pick_xy if ok else None
        return ok

    task = TaskStateMachine(
        commands=cmd, say=say, at_home=at_home, go_home=go_home,
        place_object=place_object, track_step=track_step,
        pick_object=pick_object, running=lambda: True,
        arm=arm, world=world)

    def put_object():
        ok = _put_down(arm, last_pick_xy[0])
        last_pick_xy[0] = None
        # The put-down left the arm low over the table; the tracker's last
        # command is from the hover before the pick.
        tracker.reset_command()
        return ok

    task.on_put_object = put_object

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
    mode_seen = None
    paused = False
    track_on = False
    last_look = 0.0

    while True:
        # Keep the servoj stream alive: a real controller expects a command
        # every cycle, unlike the simulator where stepping is an internal call.
        # command_step applies the state-space correction when attached.
        arm.command_step(held_q, grip)

        key = _poll_key()
        if key in ('q', 'Q', '\x1b'):
            break
        if key in ('h', 'H'):
            arm.move(cfg.UR_HOME_Q, GRIP_OPEN, rate=RAPID)
            tracker.reset_command()
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
            tracker.reset_command()

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
