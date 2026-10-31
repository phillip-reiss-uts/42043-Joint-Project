# One pick-and-place attempt, end to end.
#
# Scene layout (spawning, the chute, the object pool) lives here, but the robot
# reaches the world only through src.world.ObjectWorld. See src/world.py for
# the boundary and tests/test_layering.py for the rule that enforces it.

import mujoco
import numpy as np

from src import config as cfg

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, RAPID, FEED,
    TABLE_HEIGHT, TABLE_CENTRE, TABLE_HALF, OBJECT_HEIGHT,
    SPAWN_X_RANGE, SPAWN_Y_RANGE, SPAWN_CLEARANCE, SPAWN_ARM_CLEARANCE,
    HARD_SPAWNS, HARD_SPAWN_MODE, REALLY_HARD_MODE, HARD_BLIND_Y,
    RANDOMIZE_SPAWN, RANDOMIZE_YAW,
    MIN_OBJECT_SEPARATION,
    APPROACH_Z, LIFT_Z, GRASP_TIP_OFFSET,
    STEPS_RELEASE, DROP_HOLD_STEPS,
    SYNC_STEPS,
    MAX_PASSES,
    PLACE_EDGE_MARGIN,
    INITIAL_OBJECTS, PARK_POS,
    DROP_RADIUS,
)

from src.overlay import push
from src.pick import grasp_and_lift, search_for_object
from src.robot_backend_mujoco import gravity_feedforward


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
            _hold_arm(model, data, q_home, GRIP_OPEN)
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


def _hold_arm(model, data, q_hold, grip):
    # Command a joint hold, with the arm's gravity load fed forward.
    #
    # The compensation itself lives in the simulation backend
    # (robot_backend_mujoco.gravity_feedforward), because it models what the UR
    # controller does on hardware. These settle loops drive `data` directly
    # rather than going through the backend, so they call the same function
    # instead of keeping a second copy that could drift out of step -- a second
    # copy is exactly how the double-compensation bug arose.
    data.ctrl[:ARM_DOF] = q_hold
    data.ctrl[model.nu - 1] = grip
    gravity_feedforward(model, data, ARM_DOF)


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
                _hold_arm(model, data, q_hold, GRIP_OPEN)
                if world is not None:
                    if world is not None:
                        world.step_mover()
                mujoco.mj_step(model, data)
                while estop is not None and estop():
                    _hold_arm(model, data, q_hold, GRIP_OPEN)
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
                  stop=None, world=None, xy=None, grip_index=0):
    # Controlled release: carry the held cube to `xy`, lower it gently onto the
    # table and open the jaws.
    #
    # "Put it back down" -- the second press of the pickup key. There is no
    # chute, so a held cube is simply set down on the tabletop.
    #
    # WHERE. The caller passes the cube's current xy, so it goes back down
    # roughly where it was picked from rather than at a random clear spot (the
    # old behaviour, which was for dropping into a bin). The point is clamped
    # into the spawn region, which is the part of the table that is BOTH
    # reachable by the arm and visible to the wrist camera from home:
    #     x 0.24..0.385, y -0.075..0.145
    # Without the clamp, releasing near an edge could put the cube somewhere
    # the tracking servo can never see again -- and the next pick would fail
    # with no obvious cause.
    #
    # HOW. It lowers in two stages (high traverse, then FEED rate down) rather
    # than dropping from the lift height, so the cube is placed, not thrown.
    pos = reg.pose(data, grip_index)
    if pos[2] < TABLE_HEIGHT + OBJECT_HEIGHT:
        return False                       # not holding anything

    if xy is None:
        tx, ty = pos[0], pos[1]
    else:
        tx, ty = float(xy[0]), float(xy[1])

    # Keep the cube where the robot can find it again.
    tx = float(np.clip(tx, SPAWN_X_RANGE[0], SPAWN_X_RANGE[1]))
    ty = float(np.clip(ty, SPAWN_Y_RANGE[0], SPAWN_Y_RANGE[1]))

    # Lower with the arm's CURRENT orientation. The jaws define the cube's yaw
    # while it is held, so commanding a different one (this used to pick a
    # random yaw) would spin the cube on the way down.
    _, R = arm.kin.pose(arm.backend.joint_positions())

    # GRASP_TIP_OFFSET puts the tool at the grasp height, the same figure the
    # pick and the hardware put-down use, so releasing at this height sets the
    # cube down ON the table rather than several mm into or above it.
    release_z = TABLE_HEIGHT + OBJECT_HEIGHT / 2.0 + GRASP_TIP_OFFSET
    q_high = arm.solve_ik([tx, ty, release_z + APPROACH_Z],
                          arm.backend.joint_positions(), R_target=R)
    q_low = arm.solve_ik([tx, ty, release_z], q_high, R_target=R)
    if q_high is None or q_low is None:
        return False

    arm.move(q_high, GRIP_CLOSED, rate=RAPID, viewer=viewer, estop=estop,
             stop=stop)
    if stop is not None and stop():
        return False
    arm.move(q_low, GRIP_CLOSED, rate=FEED, viewer=viewer, estop=estop,
             stop=stop)
    if stop is not None and stop():
        return False

    arm.open_gripper(STEPS_RELEASE, viewer, estop=estop, stop=stop)

    # Let it settle on the table before anything else moves.
    q_hold = data.qpos[:ARM_DOF].copy()
    for s in range(600):
        _hold_arm(model, data, q_hold, GRIP_OPEN)
        if world is not None:
            world.step_mover()
        mujoco.mj_step(model, data)
        if viewer is not None and s % SYNC_STEPS == 0:
            viewer.sync()
    return True

def pick_one(model, data, arm, world, i, drop_xy,
             viewer=None, aborted=None, estop=None):
    # Pick pooled object i and drop it at drop_xy.
    #
    # Kept for the harness and the recovery path, which score a complete
    # pick-and-place. The interactive pickup uses pick_and_hold instead, so the
    # cube is not released until the operator asks for it.
    def stopped():
        return aborted is not None and aborted()

    ok, q_lift, R = grasp_and_lift(model, data, arm, world, i, viewer=viewer,
                           aborted=aborted, estop=estop)
    if not ok:
        return False

    # carry to the release point. No chute to clear, so this is a plain place
    # on the tabletop: lift, traverse high, then lower.
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


def clear_table(model, data, arm, world, rng,
                viewer=None, aborted=None, estop=None, passes_max=None):
    # Find the object again and put it back on the table.
    #
    # This USED to empty the table into a chute. There is no chute any more, so
    # the only thing left for it to do is recovery: after a failed grip the
    # robot re-homes and looks for the object, then re-attempts. It is no
    # longer an operator-driven mode -- run_simulation binds it to the failure
    # path of an explicit pickup.
    def stopped():
        return aborted is not None and aborted()

    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, home,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home'))
    home_q = np.array(home.qpos[:ARM_DOF])

    # Release point: a patch of tabletop short of the far edge, not a bin.
    place_cx = TABLE_CENTRE[0] + TABLE_HALF[0] - PLACE_EDGE_MARGIN
    place_cy = TABLE_CENTRE[1]

    # With no chute the table is never emptied, so this loop would otherwise run
    # to MAX_PASSES every time. The recovery caller passes passes_max to bound
    # it to a single find-and-grab attempt.
    limit = MAX_PASSES if passes_max is None else int(passes_max)
    passes = 0
    while not stopped():
        passes += 1
        if passes > limit:
            if passes_max is None:
                # With the object band left on the target never holds still, so
                # the servo never commits and each pass burns the full track
                # cap. Bound the loop rather than re-homing forever.
                print(f'[TASK] giving up after {limit} passes')
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

        drop_xy = (rng.uniform(place_cx - DROP_RADIUS, place_cx + DROP_RADIUS),
                   rng.uniform(place_cy - DROP_RADIUS, place_cy + DROP_RADIUS))
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
