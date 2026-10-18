# One pick-and-place attempt, end to end.
#
# Scene layout (spawning, the chute, the object pool) lives here, but the robot
# reaches the world only through src.world.ObjectWorld. See src/world.py for
# the boundary and tests/test_layering.py for the rule that enforces it.

import mujoco
import numpy as np

from src import config as cfg

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, RAPID, FEED, R_DOWN,
    TABLE_HEIGHT, TABLE_CENTRE, OBJECT_HEIGHT,
    SPAWN_X_RANGE, SPAWN_Y_RANGE, SPAWN_CLEARANCE, SPAWN_ARM_CLEARANCE,
    HARD_SPAWNS, HARD_SPAWN_MODE, REALLY_HARD_MODE, HARD_BLIND_Y,
    RANDOMIZE_SPAWN, RANDOMIZE_YAW,
    MIN_OBJECT_SEPARATION,
    APPROACH_Z, LIFT_Z,
    STEPS_CLOSE, STEPS_RELEASE, DROP_HOLD_STEPS,
    SERVO_V_TRANSIT, SERVO_V_DESCENT, SERVO_MAX_ITERS, SERVO_PERCEIVE_EVERY,
    SERVO_OCCLUSION_RECOVERY, SERVO_OCCLUDE_RATIO, SERVO_OCCLUDE_RECOVER,
    SERVO_OCCLUDE_FAILS, SERVO_RETREAT_STEP, SERVO_RETREAT_MAX, SERVO_STANDOFF,
    SERVO_RETREAT_V,
    REVOLVE_DEG, REVOLVE_SPEED, SYNC_STEPS,
    SEARCH_OFFSETS, SEARCH_SPEED,
    GRASP_LIFT_RECHECK,
    MAX_PASSES,
    CHUTE_X, CHUTE_Y,
    INITIAL_OBJECTS, PARK_POS,
    DROP_RADIUS,
)

from src.perception import rotation_with_yaw
from src.overlay import push
from src.servo import PBVSServo
from src.world import ObjectWorld, TargetView


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


def spawn_one(model, data, reg, rng, viewer=None, world=None):
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
            _hold_arm(model, data, q_home, GRIP_OPEN, mujoco)
            if world is not None:
                world.step_mover()  # keep the object demo advancing
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
def _hold_arm(model, data, q_hold, grip, mujoco):
    # Command a joint hold, with the arm's gravity load fed forward.
    #
    # A MuJoCo <position> servo produces kp*(ctrl - q) - kv*qvel and nothing at
    # zero error, so it cannot hold the arm against its own weight: the arm
    # accelerates, the error builds, and it oscillates and walks off. Measured
    # over 3000 steps: 1.79 rad of drift without this term, 0.000000 with it.
    #
    # Only g(q), not the full qfrc_bias: that is C(q,qdot) + g(q), and feeding
    # Coriolis back is circular -- it depends on the velocity the servo is
    # nulling, so it pumps energy in and the arm never settles (measured 0.0197
    # rad residual, qvel ~0.3 persisting). Zeroing qvel leaves exactly g(q).
    #
    # This mirrors MujocoBackend._compensate_gravity. It lives here too because
    # these settle loops drive `data` directly rather than going through the
    # backend, and a hold that drifts would undo the rest of the task.
    data.ctrl[:ARM_DOF] = q_hold
    data.ctrl[model.nu - 1] = grip
    scratch = _gravity_scratch(model, mujoco)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    data.qfrc_applied[:ARM_DOF] = scratch.qfrc_bias[:ARM_DOF]


_GRAV_SCRATCH = {}


def _gravity_scratch(model, mujoco):
    # One scratch MjData per model, so the gravity evaluation does not mutate
    # the live state and is not reallocated on every hold step.
    key = id(model)
    s = _GRAV_SCRATCH.get(key)
    if s is None:
        s = _GRAV_SCRATCH[key] = mujoco.MjData(model)
    return s


def place_cube(model, data, reg, rng, viewer=None, settle=True, index=0,
               stop=None, estop=None, world=None):
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
                _hold_arm(model, data, q_hold, GRIP_OPEN, mujoco)
                if world is not None:
                    if world is not None:
                        world.step_mover()
                mujoco.mj_step(model, data)
                while estop is not None and estop():
                    _hold_arm(model, data, q_hold, GRIP_OPEN, mujoco)
                    if world is not None:
                        world.step_mover()
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                if viewer is not None and s % SYNC_STEPS == 0:
                    viewer.sync()
        return index

    return None


def put_cube_down(model, data, arm, reg, rng, viewer=None, estop=None,
                  stop=None, world=None):
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
            _hold_arm(model, data, q_hold, GRIP_OPEN, mujoco)
            if world is not None:
                world.step_mover()  # keep the object demo advancing
            mujoco.mj_step(model, data)
            if viewer is not None and s % SYNC_STEPS == 0:
                viewer.sync()
        return True
    return False

# SCENE SETUP
def pick_one(model, data, arm, world, i, drop_xy,
             viewer=None, aborted=None, estop=None):
    # Pick pooled object i and drop it at drop_xy.
    #
    # `i` and drop_xy are scene-layout parameters chosen by the caller (a
    # simulation concern); everything the robot learns about the object comes
    # through `world`.
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
                              search=reacquire)
    if not ok or stopped():
        return False

    # Grasp committed. From here the object is held in the jaws. The mover is
    # owned by the world and keeps gliding whatever it was told to do; the
    # robot neither installs nor detaches it.
    #
    # lift from where the gripper actually ended up, with the servo's own yaw
    # estimate, so no stored target is carried past the closed-loop phase
    R = rotation_with_yaw(sv.yaw_est, R_DOWN)
    q_lift = arm.solve_ik(np.array(data.xpos[arm.base_id], float)
                          + np.array([0.0, 0.0, LIFT_Z]),
                          data.qpos[:ARM_DOF], R_target=R)

    arm.close_gripper(STEPS_CLOSE, viewer, estop=estop, stop=stopped)
    if stopped():
        return False

    # Ask the jaws whether they got anything. Without this the result is
    # inferred from the servo converging, which reports success after grasping
    # empty air -- the arm reaches its committed pose either way.
    g = arm.grasp_state()
    print(f'[GRASP] {g["state"]}: finger {g["q"]:.3f}  force {g["force"]:+.2f}  '
          f'vel {g["vel"]:+.4f}')
    push(viewer, 'gripper', 'grasp %s' % g['state'],
         'q %.3f  f %+.2f' % (g['q'], g['force']))
    if not g['held']:
        # Lift clear and read again. A first reading can be taken while the
        # fingers are still travelling, and if the cube was merely resting
        # between the jaws a lift shakes it out -- so a positive second reading
        # is the one worth trusting.
        q_recheck = arm.solve_ik(np.array(data.xpos[arm.base_id], float)
                                 + np.array([0.0, 0.0, GRASP_LIFT_RECHECK]),
                                 data.qpos[:ARM_DOF], R_target=R)
        arm.move(q_recheck, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
                 stop=stopped)
        if stopped():
            return False
        g = arm.grasp_state()
        print(f'[GRASP] recheck after lift: {g["state"]}: finger {g["q"]:.3f}  '
              f'force {g["force"]:+.2f}  vel {g["vel"]:+.4f}')
        push(viewer, 'gripper', 'grasp recheck %s' % g['state'],
             'q %.3f  f %+.2f' % (g['q'], g['force']))
        if not g['held']:
            # Nothing in the jaws. Put it down cleanly and let the caller retry
            # rather than carrying air to the chute and calling it a success.
            arm.open_gripper(STEPS_RELEASE, viewer, estop=estop, stop=stopped)
            print('[GRASP] verified empty -- abandoning this pick')
            return False

    arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             stop=stopped)

    # The old code confirmed the lift by reading the object's true height from
    # the simulator. That is not available on hardware and is not needed: the
    # jaws already reported 'held' above, from the gripper's own registers.
    # The simulation-side check lives in the harness, where ground truth
    # legitimately belongs.

    # carry to the drop point, high enough that the cube clears the chute walls
    drop_z = TABLE_HEIGHT + LIFT_Z
    q_app = arm.solve_ik([drop_xy[0], drop_xy[1], drop_z + APPROACH_Z],
                         q_lift, R_target=R)
    q_drop = arm.solve_ik([drop_xy[0], drop_xy[1], drop_z],
                          q_app, R_target=R)
    arm.move(q_app, GRIP_CLOSED, rate=RAPID, viewer=viewer, estop=estop,
             stop=stopped)
    if stopped():
        return False
    arm.move(q_drop, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             stop=stopped)
    if stopped():
        return False

    arm.open_gripper(STEPS_RELEASE, viewer, estop=estop, stop=stopped)
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
    #
    # `q` may be a full qpos vector or just the ARM_DOF joint values: solve_ik
    # returns whatever length it was given as q_init, and callers vary.
    # Assigning a 6-vector straight into qpos raised a broadcast error, so only
    # the arm joints are set here and the rest keep their defaults.
    chk = mujoco.MjData(model)
    chk.qpos[:ARM_DOF] = np.asarray(q, float)[:ARM_DOF]
    mujoco.mj_forward(model, chk)
    return float(np.linalg.norm(np.array(chk.xpos[arm.base_id])
                                - np.array(target, float))) <= tol


def clear_table(model, data, arm, world, rng,
                viewer=None, aborted=None, estop=None):
    # Pick every object on the table and drop it into the chute.
    #
    # The world supplies both perception and the object set (a layout query
    # main.py owns). No mid-motion spawning: the cube is placed by the operator
    # before a search begins.
    def stopped():
        return aborted is not None and aborted()

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

        # back to the viewing pose so the nominal view is the known-good one
        arm.move(home_q, GRIP_OPEN, rate=RAPID, viewer=viewer, estop=estop,
                 stop=stopped)
        if stopped():
            return

        def see():
            # perceive + publish to the overlay, so the annotated camera view
            # shows the same clusters the task is acting on.
            # Perception comes through the boundary: these are TargetViews.
            found = world.perceive()
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
        despawned = world.despawn_into_chute(viewer=viewer)
        if despawned:
            print(f'[TASK] despawned {despawned}')

        active = world.active()
        if not active:
            print(f'[TASK] table clear: no active objects after {passes} passes')
            return

        # The mover is owned by the world, which main.py's physics loop steps.
        # The robot neither installs nor retires it, so the object keeps
        # gliding through re-homing and searching exactly as it does during a
        # servo -- no switching on and off by the robot's state machine.

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
        # Which pooled object is being looked at is scene bookkeeping: match
        # the detection to the nearest active object by its true position.
        chosen = min(active, key=lambda j:
                     np.linalg.norm(world.pose(j)[:2] - best_xy))

        print(f'[TASK] pass {passes}: {len(active)} active, picking object '
              f'{chosen} at {np.round(best_xy, 3)} (score {best.score:.2f})')

        drop_xy = (rng.uniform(chute_cx - DROP_RADIUS, chute_cx + DROP_RADIUS),
                   rng.uniform(chute_cy - DROP_RADIUS, chute_cy + DROP_RADIUS))
        if pick_one(model, data, arm, world, chosen, drop_xy,
                    viewer=viewer, aborted=aborted, estop=estop):
            # let it fall, then despawn whatever reached the bin.
            q_hold = arm.hold_pose()
            for _ in range(3000):
                arm.hold(GRIP_OPEN, q_hold)
                world.step_mover()     # one physics tick for the object demo
                mujoco.mj_step(model, data)
                arm.observe()          # keep the estimate fresh for the next pick
                if viewer is not None and _ % SYNC_STEPS == 0:
                    viewer.sync()
                if stopped():
                    break
                while estop is not None and estop():
                    if stopped():
                        break
                    arm.hold(GRIP_OPEN, q_hold)
                    if world is not None:
                        world.step_mover()
                    mujoco.mj_step(model, data)
                    arm.observe()
                    if viewer is not None:
                        viewer.sync()
            world.despawn_into_chute(viewer=viewer)

        # hold the fingers open after the drop so the release does not blur
        q_hold = arm.hold_pose()
        for _ in range(DROP_HOLD_STEPS):
            if stopped():
                return
            arm.hold(GRIP_OPEN, q_hold)
            world.step_mover()
            mujoco.mj_step(model, data)
            arm.observe()
            if viewer is not None and _ % SYNC_STEPS == 0:
                viewer.sync()
            while estop is not None and estop():
                if stopped():
                    return
                arm.hold(GRIP_OPEN, q_hold)
                world.step_mover()
                mujoco.mj_step(model, data)
                arm.observe()
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
