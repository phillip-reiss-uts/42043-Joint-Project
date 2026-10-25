#!/usr/bin/env python3
"""Hardware front-end: UR3e + OnRobot RG2 + D435i, no simulator.

WHAT THIS IS
------------
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
teach pendant's own stop is the authority. Ctrl-C stops the console but does NOT
necessarily stop a `servoj` stream that is already running -- press the pendant
stop first. Speed is deliberately slow (FEED); raise it only after the geometry
checks pass.
"""

import argparse
import os
import sys

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
from src.task import Commands, TaskStateMachine
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

    def __init__(self, world):
        self.world = world
        self.last = []

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


def _fixed_camera_transform():
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
    arm.open_gripper(STEPS_RELEASE)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog='run_hardware.py',
                                 description='Run the pick on a real UR3e.')
    ap.add_argument('--ip', default=cfg.UR_IP, help='controller IP')
    ap.add_argument('--no-gripper', action='store_true',
                    help='arm only; grasp checks report "unknown"')
    args = ap.parse_args(argv)

    print()
    print('=' * 74)
    print('HARDWARE  UR3e + OnRobot RG2 + D435i')
    print('=' * 74)
    print('  controller  : %s' % args.ip)
    print('  cycle       : %.1f Hz (servoj period %.3f s)'
          % (1.0 / cfg.UR_CYCLE_DT, cfg.UR_CYCLE_DT))
    print('  the pendant stop is the only e-stop; speed starts at FEED')
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
    if cfg.CAMERA_MODE == 'fixed':
        T_base_cam = _fixed_camera_transform()
        world = VisibleWorld(FixedCameraWorld(depth, flange_pose=flange_pose,
                                              T_base_cam=T_base_cam))
        print('[HW] camera: fixed (eye-to-hand) at %s, aimed at %s'
              % (np.round(cfg.FIXED_CAM_POS, 3),
                 np.round(cfg.FIXED_CAM_TARGET, 3)))
    else:
        world = VisibleWorld(RealObjectWorld(depth, flange_pose=flange_pose,
                                             T_flange_cam=_camera_mount()))
        print('[HW] camera: wrist-mounted (eye-in-hand)')

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

    # ---- operator console ------------------------------------------------
    print()
    print('  p = pick up / put down (toggle)')
    print('  h = go home          s = status          q = quit')
    print()
    print('[STATE] WAIT  (place the cube, then press p)')

    held_q = np.array(backend.joint_positions(), float)
    grip = GRIP_OPEN
    mode_seen = None

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
            world.refresh()
            cmd.pickup = True
        elif key in ('s', 'S'):
            p, _ = kin.pose(backend.joint_positions())
            print('[STATE] %s  holding=%s  tool=(%.3f, %.3f, %.3f)'
                  % (task.mode, task.holding, p[0], p[1], p[2]))
        elif key in ('\r', '\n', ' '):
            pass

        # Advance the policy. This is where a pick runs; it blocks until the
        # grasp (or the put-down) finishes, streaming servoj targets itself.
        try:
            task.step()
        except KeyboardInterrupt:
            print('\n[HW] interrupted -- the arm holds its last target')
            continue

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
