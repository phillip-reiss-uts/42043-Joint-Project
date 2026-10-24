"""The robot-side pick sequence: find, approach, grasp, lift, carry.

WHY THIS IS NOT IN pick_place.py
--------------------------------
`pick_place.py` is the SIMULATION scene: it composes the table, spawns the
object pool, teleports cubes and scores their true poses. It imports mujoco, and
it should -- none of that exists on a real robot.

The pick sequence itself is robot-side logic. It only ever sees the robot
(`arm`), the boundary (`ObjectWorld`), and camera-derived `TargetView` records,
so it must run unchanged on the UR3e. Keeping it here is what lets
`run_hardware.py` import it without dragging MuJoCo onto the robot laptop.

The only simulation-flavoured arguments are `model`/`data`, kept so existing
call sites do not have to change; **they are unused** -- everything comes from
`arm.kin`, `arm.backend` and `world`. Passing None is supported and is what the
hardware front-end does.
"""

import numpy as np

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, R_DOWN, FEED, LIFT_Z,
    SPAWN_X_RANGE, SPAWN_Y_RANGE,
    SERVO_V_TRANSIT, SERVO_V_DESCENT, SERVO_MAX_ITERS, SERVO_PERCEIVE_EVERY,
    SERVO_OCCLUSION_RECOVERY, SERVO_OCCLUDE_RATIO, SERVO_OCCLUDE_RECOVER,
    SERVO_OCCLUDE_FAILS, SERVO_RETREAT_STEP, SERVO_RETREAT_MAX,
    SERVO_STANDOFF, SERVO_RETREAT_V,
    REVOLVE_DEG, REVOLVE_SPEED, SEARCH_OFFSETS, SEARCH_SPEED,
    GRASP_LIFT_RECHECK, STEPS_CLOSE, STEPS_RELEASE,
)
from src.perception import rotation_with_yaw
from src.servo import PBVSServo, push


def search_for_object(model, data, arm, perceive, viewer=None, estop=None,
                      stop=None, anchor=None, wide=False):
    # Sweep the blind strip, then revolve, looking for the object. `perceive`
    # is the caller's own perception callable, so the task-level search and the
    # servo's re-acquire are one implementation rather than two.
    if anchor is None:
        anchor = np.array(arm.kin.pose(arm.backend.joint_positions())[0], float)
    anchor = np.array(anchor, float)
    z = float(anchor[2])

    def sweep(target, rate, rot=None, follow=True):
        # Move to one search pose and look. Returns detections, or [] to keep
        # sweeping. `follow=False` means keep the gripper where it is and only
        # turn the wrist.
        if follow:
            q = arm.solve_ik(target, arm.backend.joint_positions(),
                             R_target=rot if rot is not None else R_DOWN)
        else:
            q = arm.solve_ik(target, arm.backend.joint_positions(), R_target=rot)
        if q is None or not _ik_reaches(arm, q, target):
            return []
        arm.move(q, GRIP_OPEN, rate=rate, viewer=viewer, estop=estop, stop=stop)
        if stop is not None and stop():
            return None
        return perceive()

    # Wide re-acquisition first: a target that slipped out of the footprint
    # while the arm was chasing can be far outside the small offsets below, and
    # the narrow sweep alone never finds it. Coarse grid over the spawn region.
    if wide:
        for y in np.linspace(SPAWN_Y_RANGE[0], SPAWN_Y_RANGE[1], 3):
            for x in np.linspace(SPAWN_X_RANGE[0], SPAWN_X_RANGE[1], 3):
                dets = sweep([x, y, z], SEARCH_SPEED)
                if dets is None:
                    return []
                if dets:
                    return dets

    for dx, dy in SEARCH_OFFSETS:
        dets = sweep([anchor[0] + dx, anchor[1] + dy, z], SEARCH_SPEED)
        if dets is None:
            return []
        if dets:
            return dets

    for ang in REVOLVE_DEG:
        R_ang = rotation_with_yaw(ang, R_DOWN)
        dets = sweep([anchor[0], anchor[1], z], REVOLVE_SPEED, rot=R_ang,
                     follow=False)
        if dets is None:
            return []
        if dets:
            return dets
    return []


def _ik_reaches(arm, q, target, tol=1e-3):
    # Reject a search pose the IK could not actually reach, so the sweep does
    # not command a move that goes somewhere else entirely.
    #
    # `q` may be a full qpos vector or just the ARM_DOF joint values: solve_ik
    # returns whatever length it was given as q_init, and callers vary.
    #
    # FK comes from the arm's Kinematics provider, so this works on hardware
    # where there is no MuJoCo model to forward.
    p, _ = arm.kin.pose(np.asarray(q, float)[:ARM_DOF])
    return float(np.linalg.norm(np.asarray(p, float)
                                - np.asarray(target, float))) <= tol


def grasp_and_lift(model, data, arm, world, i, viewer=None, aborted=None,
                   estop=None):
    # Approach, close, verify and lift. Returns (ok, q_lift, R).
    #
    # `i` is scene bookkeeping only; everything the robot learns about the
    # object comes through `world`. Where the cube goes afterwards is the
    # caller's decision -- pick_one releases it on the table, pick_and_hold
    # carries it to a hold pose and keeps it in the jaws.
    def stopped():
        return aborted is not None and aborted()

    # Closed-loop approach: the servo drives the WHOLE transit, re-observing and
    # re-deriving its target every perceive_every iterations. coarse=False means
    # no open-loop preposition, so vision corrects the entire approach rather
    # than only the final descent. Everything after the grasp stays open loop,
    # which the overlay shows as the trail turning red.
    sv = PBVSServo(model, data, arm, world, smooth=1.0,
                   perceive_every=SERVO_PERCEIVE_EVERY,
                   v_max=SERVO_V_TRANSIT, v_descent=SERVO_V_DESCENT,
                   occlude_ratio=SERVO_OCCLUDE_RATIO,
                   occlude_recover=SERVO_OCCLUDE_RECOVER,
                   occlude_fails=SERVO_OCCLUDE_FAILS,
                   retreat_step=SERVO_RETREAT_STEP,
                   retreat_max=SERVO_RETREAT_MAX,
                   retreat_v=SERVO_RETREAT_V)
    if not sv.observe():
        return False, None, None

    def reacquire():
        # Handed to the servo: when the target leaves the view entirely, run
        # the same sweep the task uses and report whether it found anything.
        def perceive():
            return [True] if sv.observe() else []
        return search_for_object(model, data, arm, perceive, viewer=viewer,
                                 estop=estop, stop=stopped, wide=True)

    ok, _, _, _ = sv.approach(standoff=SERVO_STANDOFF, viewer=viewer,
                              estop=estop,
                              aborted=stopped, coarse=False,
                              max_iters=SERVO_MAX_ITERS,
                              check_occlusion=SERVO_OCCLUSION_RECOVERY,
                              recover=SERVO_OCCLUSION_RECOVERY,
                              search=reacquire)
    if not ok or stopped():
        return False, None, None

    # Grasp committed. From here the object is held in the jaws. The mover is
    # owned by the world and keeps gliding whatever it was told to do; the
    # robot neither installs nor detaches it.
    #
    # Lift from where the gripper actually ended up, with the servo's own yaw
    # estimate, so no stored target is carried past the closed-loop phase.
    R = rotation_with_yaw(sv.yaw_est, R_DOWN)
    q_lift = arm.solve_ik(np.array(world.gripper_pose()[0], float)
                          + np.array([0.0, 0.0, LIFT_Z]),
                          arm.backend.joint_positions(), R_target=R)

    arm.close_gripper(STEPS_CLOSE, viewer, estop=estop, stop=stopped)
    if stopped():
        return False, None, None

    # Ask the jaws whether they got anything -- but only when the gripper can
    # actually answer. Without this the result is inferred from the servo
    # converging, which reports success after grasping empty air: the arm
    # reaches its committed pose either way.
    #
    # On this cell the RG2 has no force readback, so grasp_state() returns
    # 'unknown'. The pickup is therefore OPEN LOOP by design (todo B5):
    # 'unknown' means there is no evidence either way, so it proceeds; only an
    # explicit 'empty' abandons the pick. With a gripper that reports force --
    # the simulator, a Robotiq -- behaviour is unchanged.
    g = arm.grasp_state()
    print(f'[GRASP] {g["state"]}: finger {g["q"]:.3f}  force {g["force"]:+.2f}  '
          f'vel {g["vel"]:+.4f}')
    push(viewer, 'gripper', 'grasp %s' % g['state'],
         'q %.3f  f %+.2f' % (g['q'], g['force']))

    # Advisory only, never a gate (todo C3): the gripper can sometimes say
    # whether its jaws met an object (RG2 XML-RPC rg_get_grip_detected). It is
    # logged because it is free evidence at the bench; the pickup proceeds
    # regardless of the answer. Backends without the capability return None and
    # this prints nothing.
    _detect = getattr(arm.backend, 'grip_detected', None)
    if _detect is not None:
        try:
            detected = _detect()
        except Exception:
            detected = None
        if detected is not None:
            print('[GRASP] gripper reports %s'
                  % ('an object' if detected else 'nothing between the jaws'))
            push(viewer, 'gripper', 'detected %s'
                 % ('object' if detected else 'nothing'))

    if g['state'] == 'unknown':
        print('[GRASP] no gripper feedback -- open loop, carrying on')
    elif not g['held']:
        # Lift clear and read again. A first reading can be taken while the
        # fingers are still travelling, and if the cube was merely resting
        # between the jaws a lift shakes it out -- so a positive second reading
        # is the one worth trusting.
        q_recheck = arm.solve_ik(np.array(world.gripper_pose()[0], float)
                                 + np.array([0.0, 0.0, GRASP_LIFT_RECHECK]),
                                 arm.backend.joint_positions(), R_target=R)
        arm.move(q_recheck, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
                 stop=stopped)
        if stopped():
            return False, None, None
        g = arm.grasp_state()
        print(f'[GRASP] recheck after lift: {g["state"]}: finger {g["q"]:.3f}  '
              f'force {g["force"]:+.2f}  vel {g["vel"]:+.4f}')
        push(viewer, 'gripper', 'grasp recheck %s' % g['state'],
             'q %.3f  f %+.2f' % (g['q'], g['force']))
        # Feedback can disappear mid-pick; that is 'unknown', not 'empty'.
        if g['state'] != 'unknown' and not g['held']:
            # Nothing in the jaws. Put it down cleanly and let the caller retry
            # rather than carrying air to the release point and calling it a
            # success.
            arm.open_gripper(STEPS_RELEASE, viewer, estop=estop, stop=stopped)
            print('[GRASP] verified empty -- abandoning this pick')
            return False, None, None

    arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             stop=stopped)

    # The old code confirmed the lift by reading the object's true height from
    # the simulator. That is not available on hardware and is not needed: the
    # jaws already reported 'held' above, from the gripper's own registers.
    return not stopped(), q_lift, R


def pick_and_hold(model, data, arm, world, i, q_hold, viewer=None,
                  aborted=None, estop=None):
    # Grasp the object, lift, then carry it to `q_hold` with the jaws closed.
    #
    # The cube is NOT released. That is the point: the operator's pickup command
    # means "take it", not "take it and immediately put it back". The cube stays
    # in the jaws until the pickup command is toggled off, at which point the
    # caller runs the put-down (see src/task.py HOLD/PUTDOWN).
    def stopped():
        return aborted is not None and aborted()

    ok, q_lift, R = grasp_and_lift(model, data, arm, world, i, viewer=viewer,
                                   aborted=aborted, estop=estop)
    if not ok:
        return False
    arm.move(q_hold, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             stop=stopped)
    if stopped():
        return False
    push(viewer, 'gripper', 'holding at home',
         'press the pickup key to release')
    return True
