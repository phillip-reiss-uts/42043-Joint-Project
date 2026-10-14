# One pick-and-place attempt, end to end.

# DISABLED: dataclass was only used by Result, which is commented out above.
# from dataclasses import dataclass

import mujoco
import numpy as np

from src import config as cfg

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, RAPID, FEED, R_DOWN,
    TABLE_HEIGHT, TABLE_CENTRE, OBJECT_HEIGHT,
    SPAWN_X_RANGE, SPAWN_Y_RANGE, SPAWN_CLEARANCE, SPAWN_ARM_CLEARANCE,
    HARD_SPAWNS, HARD_SPAWN_MODE, REALLY_HARD_MODE, HARD_BLIND_Y,
    RANDOMIZE_SPAWN, RANDOMIZE_YAW,
    MIN_OBJECT_POINTS, MIN_OBJECT_SEPARATION,
    TIP_OFFSET, APPROACH_Z, FINAL_Z, LIFT_Z,
    PLACE_DX, PLACE_DY, PLACE_OFFSET_X, PLACE_OFFSET_Y,
    PLACE_CLEARANCE, STEPS_CLOSE, STEPS_RELEASE, DROP_HOLD_STEPS,
    SERVO_V_TRANSIT, SERVO_V_DESCENT, SERVO_MAX_ITERS, SERVO_PERCEIVE_EVERY,
    SERVO_OCCLUSION_RECOVERY, SERVO_OCCLUDE_RATIO, SERVO_OCCLUDE_RECOVER,
    SERVO_OCCLUDE_FAILS, SERVO_RETREAT_STEP, SERVO_RETREAT_MAX, SERVO_STANDOFF,
    SERVO_RETREAT_V,
    REVOLVE_DEG, REVOLVE_SPEED, SYNC_STEPS,
    SEARCH_OFFSETS, SEARCH_SPEED,
    MAX_PASSES,
    # LEFTOVER: POOL_SIZE is passed in by main.py; this module never reads it.
    CHUTE_X, CHUTE_Y,
    INITIAL_OBJECTS, PARK_POS,
    DROP_RADIUS,
    # LEFTOVER: SYNC_DT is unused here; the rest of the block was unused too.
)

from src.perception import perceive, perceive_all, rotation_with_yaw
from src.overlay import push
from src.servo import PBVSServo
from src.target_motion import TargetMover


# @dataclass
# --- DISABLED: Result, only ever constructed inside the stale path above. ---
# class Result:
#     # Outcome of one attempt. The batch harness reads these fields.
#     found: bool = False         # perception produced a usable object
#     picked: bool = False        # object left the table
#     placed: bool = False        # object settled back on the surface
#     yaw: float = 0.0
#     perr: float = float('nan')  # perception error vs truth (simulation only)
#     placed_err: float = float('nan')
#     n_samples: int = 0          # resamples consumed before perception succeeded
#     n_peeks: int = 0
#     aborted: bool = False
#
#
def _log(msg):
    # Stage logging, gated at call time so config.VERBOSE can be toggled.
    if cfg.VERBOSE:
        print(msg)


# OBJECT POOL
class ObjectRegistry:
    # Pool objects and their per-object state.

    def __init__(self, model, pool_size):
        self.model = model
        self.body_ids = []
        self.joint_ids = []
        self.qpos_adrs = []
        self.qvel_adrs = []
        self.state = []       # 'parked' | 'active' | 'chuted'

        for i in range(pool_size):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'obj_{i}')
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                      f'obj_{i}_free')
            self.body_ids.append(body)
            self.joint_ids.append(joint)
            self.qpos_adrs.append(model.jnt_qposadr[joint])
            self.qvel_adrs.append(model.jnt_dofadr[joint])
            self.state.append('parked')

    def active(self):
        return [i for i in range(len(self.body_ids)) if self.state[i] == 'active']

    def parked(self):
        return [i for i in range(len(self.body_ids)) if self.state[i] == 'parked']

    def pose(self, data, i):
        # Object centre from qpos (freejoint stores the body position).
        adr = self.qpos_adrs[i]
        return np.array(data.qpos[adr:adr + 3]).copy()

    def freeze(self, data, i):
        # Zero the object's velocity -- despawned or parked objects stay put.
        adr = self.qvel_adrs[i]
        data.qvel[adr:adr + 6] = 0.0

    def teleport(self, data, i, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        # Instantly move object i. Caller decides whether to settle or freeze.
        adr = self.qpos_adrs[i]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = quat
        self.freeze(data, i)


def _separation_ok(reg, data, cand_xy, active_only=True):
    # True if cand_xy is MIN_OBJECT_SEPARATION from every active object.
    for i in reg.active() if active_only else range(len(reg.body_ids)):
        if reg.state[i] not in ('active',):
            continue
        if np.linalg.norm(reg.pose(data, i)[:2] - np.asarray(cand_xy)) \
                < MIN_OBJECT_SEPARATION:
            return False
    return True


def _choose_spawn_xy(rng, fallback=False):
    # Pick a spawn xy according to the configured spawn mode.
    if fallback:
        return rng.uniform(*SPAWN_X_RANGE), rng.uniform(*SPAWN_Y_RANGE)
    if REALLY_HARD_MODE:
        return rng.uniform(*SPAWN_X_RANGE), rng.uniform(*HARD_BLIND_Y)
    if HARD_SPAWN_MODE:
        x, y = HARD_SPAWNS[int(rng.integers(len(HARD_SPAWNS)))]
        return float(x), float(y)
    if RANDOMIZE_SPAWN:
        return rng.uniform(*SPAWN_X_RANGE), rng.uniform(*SPAWN_Y_RANGE)
    return TABLE_CENTRE[0], TABLE_CENTRE[1]


def spawn_one(model, data, reg, rng, viewer=None):
    # Activate one parked object at a separation-checked pose.
    parked = reg.parked()
    if not parked:
        return None

    i = parked[0]
    for k in range(60):
        # prefer the configured (hard) positions, then fall back to the easy band
        x, y = _choose_spawn_xy(rng, fallback=(k >= 30))
        cand = np.array([x, y])
        if not _separation_ok(reg, data, cand):
            continue

        yaw = rng.uniform(0.0, np.pi / 2.0) if RANDOMIZE_YAW else 0.0
        reg.teleport(data, i,
                     [x, y, TABLE_HEIGHT + SPAWN_CLEARANCE],
                     quat=(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        reg.state[i] = 'active'
        reg.freeze(data, i)
        mujoco.mj_forward(model, data)

        # settle: let it fall onto the table, sync the viewer through the fall
        q_home = data.qpos[:ARM_DOF].copy()
        for s in range(2000):
            data.ctrl[:ARM_DOF] = q_home
            data.ctrl[model.nu - 1] = GRIP_OPEN
            mujoco.mj_step(model, data)
            if viewer is not None and s < 200 and s % SYNC_STEPS == 0:
                viewer.sync()
        if viewer is not None:
            viewer.sync()
        return i

    return None


def spawn_now(model, data, reg, rng, viewer=None):
    # Drop a parked object onto the table immediately -- no settle loop.
    parked = reg.parked()
    if not parked:
        return None

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    base_xy = np.array(data.xpos[base_id][:2], float)

    i = parked[0]
    for k in range(120):
        # hard positions first, easy band once they are exhausted or occupied
        cand = np.array(_choose_spawn_xy(rng, fallback=(k >= 60)))
        # clear of the arm/gripper
        if np.linalg.norm(cand - base_xy) < SPAWN_ARM_CLEARANCE:
            continue
        # clear of every cube already on the table
        if not _separation_ok(reg, data, cand):
            continue

        yaw = rng.uniform(0.0, np.pi / 2.0) if RANDOMIZE_YAW else 0.0
        reg.teleport(data, i,
                     [cand[0], cand[1], TABLE_HEIGHT + SPAWN_CLEARANCE],
                     quat=(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        reg.state[i] = 'active'
        reg.freeze(data, i)
        mujoco.mj_forward(model, data)
        return i

    return None


def despawn_into_chute(model, data, reg, viewer=None):
    # Freeze any active object that has fallen into the chute region.
    changed = []
    for i in reg.active():
        pos = reg.pose(data, i)
        if pos[2] < TABLE_HEIGHT - 0.02:
            # below the tabletop: only the chute floor exists down there
            reg.freeze(data, i)
            reg.state[i] = 'chuted'
            changed.append(i)
            if viewer is not None:
                viewer.sync()
    return changed


# SINGLE-CUBE COMMANDS
def place_cube(model, data, reg, rng, viewer=None, settle=True, index=0,
               stop=None, estop=None):
    # Put the cube on the table at a spawn position, whether it was parked,
    # held or already on the table. Honours the hard-spawn mode through
    # _choose_spawn_xy: HARD_SPAWN_MODE tries the measured awkward positions
    # first and falls back to the easy band.
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    base_xy = np.array(data.xpos[base_id][:2], float)

    for k in range(200):
        cand = np.array(_choose_spawn_xy(rng, fallback=(k >= 100)))
        if np.linalg.norm(cand - base_xy) < SPAWN_ARM_CLEARANCE:
            continue
        # clearance from other objects only -- the cube itself may be active
        clash = False
        for j in reg.active():
            if j == index:
                continue
            if np.linalg.norm(reg.pose(data, j)[:2] - cand) < MIN_OBJECT_SEPARATION:
                clash = True
                break
        if clash:
            continue

        yaw = rng.uniform(0.0, np.pi / 2.0) if RANDOMIZE_YAW else 0.0
        reg.teleport(data, index,
                     [cand[0], cand[1], TABLE_HEIGHT + SPAWN_CLEARANCE],
                     quat=(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        reg.state[index] = 'active'
        reg.freeze(data, index)
        mujoco.mj_forward(model, data)

        if settle:
            q_hold = data.qpos[:ARM_DOF].copy()
            for s in range(2000):
                if stop is not None and stop():
                    break
                data.ctrl[:ARM_DOF] = q_hold
                data.ctrl[model.nu - 1] = GRIP_OPEN
                mujoco.mj_step(model, data)
                while estop is not None and estop():
                    data.ctrl[:ARM_DOF] = q_hold
                    data.ctrl[model.nu - 1] = GRIP_OPEN
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                if viewer is not None and s % SYNC_STEPS == 0:
                    viewer.sync()
        return index

    return None


def put_cube_down(model, data, arm, reg, rng, viewer=None, estop=None,
                  stop=None):
    # Controlled release: carry the held cube to a clear spot on the table,
    # lower it and open the jaws. Used by the S-stop command.
    i = 0
    pos = reg.pose(data, i)
    if pos[2] < TABLE_HEIGHT + OBJECT_HEIGHT:
        return False                       # not holding anything

    for k in range(200):
        cand = np.array(_choose_spawn_xy(rng, fallback=(k >= 100)))
        R = rotation_with_yaw(float(rng.uniform(0.0, np.pi / 2.0)), R_DOWN)
        q_high = arm.solve_ik([cand[0], cand[1], TABLE_HEIGHT + LIFT_Z],
                              data.qpos[:ARM_DOF], R_target=R)
        q_low = arm.solve_ik([cand[0], cand[1], TABLE_HEIGHT + OBJECT_HEIGHT],
                             q_high, R_target=R)
        arm.move(q_high, GRIP_CLOSED, rate=RAPID, viewer=viewer, estop=estop,
                 stop=stop)
        if stop is not None and stop():
            return False
        arm.move(q_low, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
                 stop=stop)
        arm.open_gripper(STEPS_RELEASE, viewer, estop=estop, stop=stop)
        q_hold = data.qpos[:ARM_DOF].copy()
        for s in range(600):
            data.ctrl[:ARM_DOF] = q_hold
            data.ctrl[model.nu - 1] = GRIP_OPEN
            mujoco.mj_step(model, data)
            if viewer is not None and s % SYNC_STEPS == 0:
                viewer.sync()
        return True
    return False

# SCENE SETUP
# --- DISABLED: stale pre-servo pick path. Never called: pick_and_place -> find_object -> _detect_once -> place_object. Superseded by pick_one/clear_table. Kept for reference. ---
# def place_object(model, data, rng, home_id, free_joint_id, body_id, viewer=None):
#     # Reset the scene, put the object somewhere, and let it settle.
#     mujoco.mj_resetDataKeyframe(model, data, home_id)
#     mujoco.mj_forward(model, data)
#
#     if viewer is not None:
#         viewer.sync()  
#
#     q_home = data.qpos[:ARM_DOF].copy()
#     adr = model.jnt_qposadr[free_joint_id]
#
#     # spawn mode lives in _choose_spawn_xy so both paths cannot drift apart.
#     x, y = _choose_spawn_xy(rng)
#
#     data.qpos[adr:adr + 3] = [x, y, TABLE_HEIGHT + SPAWN_CLEARANCE]
#     if RANDOMIZE_YAW:
#         # cube: 90 deg symmetry, so [0, pi/2) covers every distinguishable yaw
#         yaw = rng.uniform(0.0, np.pi / 2.0)
#         data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
#     else:
#         data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
#     mujoco.mj_forward(model, data)
#
#     q_home = data.qpos[:ARM_DOF].copy()
#     for i in range(2000):
#         data.ctrl[:ARM_DOF] = q_home
#         data.ctrl[model.nu - 1] = GRIP_OPEN
#         mujoco.mj_step(model, data)
#         # the object only moves for the first ~200 steps; sync through it.
#         if viewer is not None:
#             if i < 200 and i % SYNC_STEPS  == 0:
#                 viewer.sync()
#             elif i >= 200 and i % 200 == 0:
#                 viewer.sync()
#
#     return q_home, np.array(data.xpos[body_id]).copy()
#
#
# def _detect_once(model, data, arm, camera, cam_id, rng, result, home_id,
#                  free_joint_id, body_id, viewer, aborted, respawn=True):
#     # Place the object once, then search for it.
#     def stopped():
#         return aborted is not None and aborted()
#
#     def reached(q, target):
#         # True if IK actually put gripper/base at target.
#         check = mujoco.MjData(model)
#         check.qpos[:] = q
#         mujoco.mj_forward(model, check)
#         return np.linalg.norm(np.array(check.xpos[arm.base_id])
#                               - np.array(target)) < 1e-3
#
#     if respawn:
#         q_home, truth = place_object(model, data, rng, home_id,
#                                      free_joint_id, body_id, viewer=viewer)
#     else:
#
#
#         print(f'[DBG] no-respawn branch, object at '
#               f'{np.round(data.xpos[body_id][:2], 3)}')        
#         # keep the scene, but return to the nominal viewing pose, then search.
#         truth = np.array(data.xpos[body_id]).copy()
#         d_ref = mujoco.MjData(model)
#         mujoco.mj_resetDataKeyframe(model, d_ref, home_id)
#         q_home = np.array(d_ref.qpos[:ARM_DOF])
#         arm.move(q_home, GRIP_OPEN, rate=SEARCH_SPEED, viewer=viewer)
#     if stopped():
#         result.aborted = True
#         return None, None, truth
#
#     det = perceive(model, data, camera, cam_id)
#     if det.ok and det.n_points >= MIN_OBJECT_POINTS:
#         return det, data.qpos[:ARM_DOF].copy(), truth
#     _log(f'[PERCEPTION] nominal view weak ({det.n_points} pts); searching')
#
#     # the nominal pose is the anchor: offsets are applied to it, not accumulated
#     anchor = np.array(data.xpos[arm.base_id], float)
#     tgt_z = float(anchor[2])
#
#     # 1. TRANSLATE
#     for dx, dy in SEARCH_OFFSETS:
#         target = [anchor[0] + dx, anchor[1] + dy, tgt_z]
#         q = arm.solve_ik(target, data.qpos[:ARM_DOF], R_target=R_DOWN)
#         if not reached(q, target):
#             _log(f'[PERCEPTION] search ({dx:+.2f},{dy:+.2f}): IK failed')
#             continue
#
#         arm.move(q, GRIP_OPEN, rate=SEARCH_SPEED, viewer=viewer)
#         result.n_peeks += 1
#         if stopped():
#             result.aborted = True
#             return None, None, truth
#
#         det = perceive(model, data, camera, cam_id)
#         _log(f'[PERCEPTION] search ({dx:+.2f},{dy:+.2f}) -> {det.n_points} pts')
#         if det.ok and det.n_points >= MIN_OBJECT_POINTS:
#             return det, data.qpos[:ARM_DOF].copy(), truth
#
#     # 2. REVOLVE
#     for ang in REVOLVE_DEG:
#         R_ang = rotation_with_yaw(ang, R_DOWN)
#         q = arm.solve_ik([anchor[0], anchor[1], tgt_z], data.qpos[:ARM_DOF],
#                          R_target=R_ang)
#         if not reached(q, [anchor[0], anchor[1], tgt_z]):
#             _log(f'[PERCEPTION] revolve {ang:3d} deg: IK failed')
#             continue
#
#         arm.move(q, GRIP_OPEN, rate=REVOLVE_SPEED, viewer=viewer)
#         result.n_peeks += 1
#         if stopped():
#             result.aborted = True
#             return None, None, truth
#
#         det = perceive(model, data, camera, cam_id)
#         _log(f'[PERCEPTION] revolve {ang:3d} deg -> {det.n_points} pts')
#         if det.ok and det.n_points >= MIN_OBJECT_POINTS:
#             return det, data.qpos[:ARM_DOF].copy(), truth
#
#     return None, None, truth
#
#
# def find_object(model, data, arm, camera, cam_id, rng, result,
#                 home_id, free_joint_id, body_id, viewer=None, aborted=None,
#                 respawn=True):
#     # Place the object once, then search for it.
#     det, q_plan, truth = _detect_once(
#         model, data, arm, camera, cam_id, rng, result, home_id,
#         free_joint_id, body_id, viewer, aborted, respawn=respawn)
#
#     return det, q_plan, truth
#
# # TASK
# def pick_and_place(model, data, arm, camera, cam_id, rng,
#                    viewer=None, aborted=None, respawn=True):
#     # One pick-and-place attempt.
#     result = Result()
#
#     def stopped():
#         return aborted is not None and aborted()
#
#     home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
#     free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
#     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
#
#     det, q_plan, truth = find_object(
#         model, data, arm, camera, cam_id, rng, result,
#         home_id, free_joint_id, body_id, viewer=viewer, aborted=aborted,
#         respawn=respawn)
#
#     if det is None:
#         print(f'[WARN] object not found after {result.n_samples} samples and '
#               f'{result.n_peeks} peeks')
#         return result
#
#     result.found = True
#     centre, yaw = det.centre, det.yaw
#     result.yaw = yaw
#     if truth is not None:
#         result.perr = float(np.linalg.norm(centre - truth))
#
#     _log(f'[PERCEPTION] {det.n_points} points  centre={np.round(centre, 4)}  '
#          f'yaw={yaw:.1f} deg  error={result.perr:.4f} m')
#
#     R_grasp = rotation_with_yaw(yaw, R_DOWN)
#     R_place = R_grasp
#
#     grasp_z = centre[2] + TIP_OFFSET
#     q_approach = arm.solve_ik([centre[0], centre[1], grasp_z + APPROACH_Z],
#                               q_plan, R_target=R_grasp)
#     q_final = arm.solve_ik([centre[0], centre[1], grasp_z + FINAL_Z],
#                            q_approach, R_target=R_grasp)
#     q_grasp = arm.solve_ik([centre[0], centre[1], grasp_z],
#                            q_final, R_target=R_grasp)
#     q_lift = arm.solve_ik([centre[0], centre[1], grasp_z + LIFT_Z],
#                           q_grasp, R_target=R_grasp)
#     _log(f'[PLAN] grasp base z={grasp_z:.4f}')
#
#     arm.move(q_approach, GRIP_OPEN, rate=RAPID, viewer=viewer)
#     if stopped():
#         result.aborted = True
#         return result
#     arm.move(q_final, GRIP_OPEN, rate=FEED, viewer=viewer)
#     arm.move(q_grasp, GRIP_OPEN, rate=FEED, viewer=viewer)
#     if stopped():
#         result.aborted = True
#         return result
#     _log(f'[EXEC] at grasp base={np.round(data.xpos[arm.base_id], 4)}')
#
#     arm.close_gripper(STEPS_CLOSE, viewer)
#     if stopped():
#         result.aborted = True
#         return result
#
#     arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=viewer)
#     lifted = float(data.xpos[body_id][2]) - TABLE_HEIGHT
#     result.picked = lifted > 0.06
#     _log(f'[RESULT] PICK {"OK" if result.picked else "FAILED"} '
#          f'-- object {lifted:+.4f} m above the surface')
#
#     # PLACE
#     place_xy = (TABLE_CENTRE[0] + PLACE_DX + PLACE_OFFSET_X, TABLE_CENTRE[1] + PLACE_DY + PLACE_OFFSET_Y)
#     place_base_z = grasp_z + PLACE_CLEARANCE
#
#     q_place_app = arm.solve_ik([place_xy[0], place_xy[1],
#                                 place_base_z + APPROACH_Z],
#                                q_lift, R_target=R_place)
#     q_place = arm.solve_ik([place_xy[0], place_xy[1], place_base_z],
#                            q_place_app, R_target=R_place)
#     q_retreat = arm.solve_ik([place_xy[0], place_xy[1],
#                               place_base_z + LIFT_Z],
#                              q_place, R_target=R_place)
#
#     arm.move(q_place_app, GRIP_CLOSED, rate=RAPID, viewer=viewer)
#     if stopped():
#         result.aborted = True
#         return result
#     arm.move(q_place, GRIP_CLOSED, rate=FEED, viewer=viewer)
#     if stopped():
#         result.aborted = True
#         return result
#
#     arm.open_gripper(STEPS_RELEASE, viewer)
#     if stopped():
#         result.aborted = True
#         return result
#
#     arm.move(q_retreat, GRIP_OPEN, rate=RAPID, viewer=viewer)
#
#     placed = np.array(data.xpos[body_id])
#     result.placed_err = float(np.linalg.norm(placed[:2] - np.array(place_xy)))
#     result.placed = abs(placed[2] - (TABLE_HEIGHT + OBJECT_HEIGHT / 2.0)) < 0.005
#     _log(f'[RESULT] PLACE {"OK" if result.placed else "FAILED"} '
#          f'-- {result.placed_err * 1000:.1f} mm from target')
#
#     return result
#
# # Clear table
def pick_one(model, data, arm, reg, i, drop_xy, camera=None, cam_id=-1,
             viewer=None, aborted=None, estop=None, spawn_hook=None, rng=None):
    # Pick pooled object i and drop it at drop_xy.
    def stopped():
        return aborted is not None and aborted()

    # robustness demonstration: walk the target around while we chase it. The
    # mover is owned by the arm (see Arm.mover) so it survives the whole task;
    # clear_table installs one, and this is only a fallback for direct callers.
    mover = getattr(arm, 'mover', None)
    if mover is None:
        mover = TargetMover(model, data, reg, i, rng=rng)
        arm.mover = mover

    # Closed-loop approach: the servo drives the WHOLE transit, re-observing and
    # re-deriving its target every perceive_every iterations. coarse=False means
    # no open-loop preposition, so vision corrects the entire approach rather
    # than only the final descent. Everything after the grasp stays open loop,
    # which the overlay shows as the trail turning red.
    sv = PBVSServo(model, data, arm, camera, cam_id, smooth=1.0,
                   perceive_every=SERVO_PERCEIVE_EVERY,
                   v_max=SERVO_V_TRANSIT, v_descent=SERVO_V_DESCENT,
                   occlude_ratio=SERVO_OCCLUDE_RATIO,
                   occlude_recover=SERVO_OCCLUDE_RECOVER,
                   occlude_fails=SERVO_OCCLUDE_FAILS,
                   retreat_step=SERVO_RETREAT_STEP,
                   retreat_max=SERVO_RETREAT_MAX,
                   retreat_v=SERVO_RETREAT_V)
    if not sv.observe():
        return False

    def reacquire():
        # Handed to the servo: when the target leaves the view entirely, run
        # the same sweep the task uses and report whether it found anything.
        # observe() with no 'near' hint takes the best-scoring cluster, which is
        # the right choice once we no longer know roughly where the object is.
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
                              mover=mover, search=reacquire)
    if not ok or stopped():
        return False

    # Grasp committed. Detach the driver now: from here the object is held in
    # the jaws, and continuing to command it would fight the grip. clear_table
    # re-installs a mover on its next pass.
    arm.mover = None

    # lift from where the gripper actually ended up, with the servo's own yaw
    # estimate, so no stored target is carried past the closed-loop phase
    R = rotation_with_yaw(sv.yaw_est, R_DOWN)
    q_lift = arm.solve_ik(np.array(data.xpos[arm.base_id], float)
                          + np.array([0.0, 0.0, LIFT_Z]),
                          data.qpos[:ARM_DOF], R_target=R)

    arm.close_gripper(STEPS_CLOSE, viewer, estop=estop, spawn_hook=spawn_hook,
                      stop=stopped)
    if stopped():
        return False
    arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             spawn_hook=spawn_hook, stop=stopped)
    if reg.pose(data, i)[2] < TABLE_HEIGHT + OBJECT_HEIGHT / 2.0:
        # it did not leave the table -- abort this object rather than drop air
        arm.open_gripper(STEPS_RELEASE, viewer, estop=estop,
                         spawn_hook=spawn_hook, stop=stopped)
        return False

    # carry to the drop point, high enough that the cube clears the chute walls
    drop_z = TABLE_HEIGHT + LIFT_Z
    q_app = arm.solve_ik([drop_xy[0], drop_xy[1], drop_z + APPROACH_Z],
                         q_lift, R_target=R)
    q_drop = arm.solve_ik([drop_xy[0], drop_xy[1], drop_z],
                          q_app, R_target=R)
    arm.move(q_app, GRIP_CLOSED, rate=RAPID, viewer=viewer, estop=estop,
             spawn_hook=spawn_hook, stop=stopped)
    if stopped():
        return False
    arm.move(q_drop, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             spawn_hook=spawn_hook, stop=stopped)
    if stopped():
        return False

    arm.open_gripper(STEPS_RELEASE, viewer, estop=estop, spawn_hook=spawn_hook,
                     stop=stopped)
    return True


def search_for_object(model, data, arm, perceive, viewer=None, estop=None,
                      stop=None, anchor=None, wide=False):
    # Sweep the blind strip, then revolve, looking for the object. `perceive`
    # is the caller's own perception callable, so the task-level search and the
    # servo's re-acquire are one implementation rather than two.
    if anchor is None:
        anchor = np.array(data.xpos[arm.base_id], float)
    anchor = np.array(anchor, float)
    z = float(anchor[2])

    def sweep(target, rate, rot=None, follow=True):
        # Move to one search pose and look. Returns detections, or [] to keep
        # sweeping. `follow=False` means keep the gripper where it is and only
        # turn the wrist.
        if follow:
            q = arm.solve_ik(target, data.qpos[:ARM_DOF],
                             R_target=rot if rot is not None else R_DOWN)
        else:
            q = arm.solve_ik(target, data.qpos[:ARM_DOF], R_target=rot)
        if q is None or not _ik_reaches(model, arm, q, target):
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


def _ik_reaches(model, arm, q, target, tol=1e-3):
    # Reject a search pose the IK could not actually reach, so the sweep does
    # not command a move that goes somewhere else entirely.
    chk = mujoco.MjData(model)
    chk.qpos[:] = q
    mujoco.mj_forward(model, chk)
    return float(np.linalg.norm(np.array(chk.xpos[arm.base_id])
                                - np.array(target, float))) <= tol


def clear_table(model, data, arm, camera, cam_id, rng, reg,
                viewer=None, aborted=None, spawn_request=None, estop=None):
    # Pick every object on the table and drop it into the chute.
    def stopped():
        return aborted is not None and aborted()

    def spawn_hook():
        # Called every step by Arm.move/_ramp_grip. Non-blocking.
        if spawn_request is not None and spawn_request():
            idx = spawn_now(model, data, reg, rng)
            if idx is None:
                print('[TASK] spawn skipped: no clear space on the table')
            else:
                print(f'[TASK] dropped object {idx} mid-motion')

    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, home,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home'))
    home_q = np.array(home.qpos[:ARM_DOF])

    chute_cx = 0.5 * (CHUTE_X[0] + CHUTE_X[1])
    chute_cy = 0.5 * (CHUTE_Y[0] + CHUTE_Y[1])

    passes = 0
    while not stopped():
        passes += 1
        if passes > MAX_PASSES:
            # With the object band left on the target never holds still, so the
            # servo never commits and each pass burns the full track cap. Bound
            # the loop rather than re-homing forever.
            print(f'[TASK] giving up after {MAX_PASSES} passes')
            return

        # also service a queued spawn here, so P works while the arm is idle
        spawn_hook()

        # back to the viewing pose so the nominal view is the known-good one
        arm.move(home_q, GRIP_OPEN, rate=RAPID, viewer=viewer, estop=estop,
                 spawn_hook=spawn_hook, stop=stopped)
        if stopped():
            return

        def see():
            # perceive + publish to the overlay, so the annotated camera view
            # shows the same clusters the task is acting on
            found = perceive_all(model, data, camera, cam_id)
            ov = getattr(viewer, 'overlay', None)
            if ov is not None:
                ov.set_detections(found)
            span = getattr(found[0], 'span', None) if found else None
            push(viewer, 'perception',
                 'dets %d' % len(found),
                 'span %.4f %.4f' % (span[0], span[1]) if span is not None
                 else 'span -')
            return found

        dets = see()
        despawned = despawn_into_chute(model, data, reg, viewer=viewer)
        if despawned:
            print(f'[TASK] despawned {despawned}')

        active = reg.active()
        if not active:
            print(f'[TASK] table clear: no active objects after {passes} passes')
            arm.mover = None
            return

        # The target keeps moving for the whole task, not only while a servo is
        # running. With the mover owned by pick_one it froze solid the moment a
        # pick failed and the arm went home or began searching, then started
        # again on the next attempt -- which reads as the object being switched
        # on and off by the robot's own state machine.
        if arm.mover is None or arm.mover.index != active[0]:
            arm.mover = TargetMover(model, data, reg, active[0], rng=rng)

        # Nothing visible but objects remain: sweep the blind strip.
        if not dets:
            dets = search_for_object(model, data, arm, see, viewer=viewer,
                                     estop=estop, stop=stopped)
            if stopped():
                return
            if not dets:
                print(f'[TASK] {len(active)} active, none visible after search; '
                      f'passes={passes}')
                return

        best = dets[0]
        best_xy = np.array(best.centre[:2])
        chosen = min(active, key=lambda j:
                     np.linalg.norm(reg.pose(data, j)[:2] - best_xy))

        print(f'[TASK] pass {passes}: {len(active)} active, picking object '
              f'{chosen} at {np.round(best_xy, 3)} (score {best.score:.2f})')

        drop_xy = (rng.uniform(chute_cx - DROP_RADIUS, chute_cx + DROP_RADIUS),
                   rng.uniform(chute_cy - DROP_RADIUS, chute_cy + DROP_RADIUS))
        if pick_one(model, data, arm, reg, chosen, drop_xy,
                    camera=camera, cam_id=cam_id,
                    viewer=viewer, aborted=aborted, estop=estop,
                    spawn_hook=spawn_hook, rng=rng):
            # let it fall, then despawn whatever reached the bin.
            for _ in range(3000):
                arm.hold(GRIP_OPEN)
                mujoco.mj_step(model, data)
                arm.observe()          # keep the estimate fresh for the next pick
                spawn_hook()
                if viewer is not None and _ % SYNC_STEPS == 0:
                    viewer.sync()
                if stopped():
                    break
                while estop is not None and estop():
                    if stopped():
                        break
                    arm.hold(GRIP_OPEN)
                    mujoco.mj_step(model, data)
                    arm.observe()
                    spawn_hook()
                    if viewer is not None:
                        viewer.sync()
            despawn_into_chute(model, data, reg, viewer=viewer)

        # hold the fingers open after the drop so the release does not blur
        for _ in range(DROP_HOLD_STEPS):
            if stopped():
                return
            arm.hold(GRIP_OPEN)
            mujoco.mj_step(model, data)
            arm.observe()
            spawn_hook()
            if viewer is not None and _ % SYNC_STEPS == 0:
                viewer.sync()
            while estop is not None and estop():
                if stopped():
                    return
                arm.hold(GRIP_OPEN)
                mujoco.mj_step(model, data)
                arm.observe()
                spawn_hook()
                if viewer is not None:
                    viewer.sync()

# scene reset 
def reset_scene(model, data, reg, rng, viewer=None):
    # Park everything and spawn the initial objects. R-press behaviour.
    for i in range(len(reg.body_ids)):
        reg.teleport(data, i, PARK_POS)
        reg.state[i] = 'parked'

    spawned = []
    for _ in range(INITIAL_OBJECTS):
        idx = spawn_one(model, data, reg, rng, viewer=viewer)
        if idx is not None:
            spawned.append(idx)
        else:
            break
    return spawned
