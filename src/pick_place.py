"""One pick-and-place attempt, end to end.

Owns scene reset and object placement, the perception retry policy, and the
grasp/place sequence. Knows nothing about the viewer loop.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from src import config as cfg
from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, RAPID, FEED, R_DOWN,
    TABLE_HEIGHT, TABLE_CENTRE, OBJECT_HEIGHT,
    SPAWN_X_RANGE, SPAWN_Y_RANGE, SPAWN_CLEARANCE,
    HARD_SPAWNS, HARD_SPAWN_MODE, REALLY_HARD_MODE, HARD_BLIND_Y,
    RANDOMIZE_SPAWN, RANDOMIZE_YAW,
    MIN_OBJECT_POINTS,  
    TIP_OFFSET, APPROACH_Z, FINAL_Z, LIFT_Z,
    PLACE_DX, PLACE_DY, PLACE_OFFSET_X, PLACE_OFFSET_Y,
    PLACE_CLEARANCE, STEPS_CLOSE, STEPS_RELEASE,
    REVOLVE_DEG, REVOLVE_SPEED, SYNC_DT, SYNC_STEPS,
    SEARCH_OFFSETS, SEARCH_SPEED, REVOLVE_DEG, REVOLVE_SPEED,
)
from src.perception import perceive, rotation_with_yaw


@dataclass
class Result:
    """Outcome of one attempt. The batch harness reads these fields."""
    found: bool = False         # perception produced a usable object
    picked: bool = False        # object left the table
    placed: bool = False        # object settled back on the surface
    yaw: float = 0.0
    perr: float = float('nan')  # perception error vs truth (simulation only)
    placed_err: float = float('nan')
    n_samples: int = 0          # resamples consumed before perception succeeded
    n_peeks: int = 0
    aborted: bool = False


def _log(msg):
    """Stage logging, gated at call time so config.VERBOSE can be toggled."""
    if cfg.VERBOSE:
        print(msg)


# ------------------------------------------------------------ scene setup ---
def place_object(model, data, rng, home_id, free_joint_id, body_id, viewer=None):
    """Reset the scene, put the object somewhere, and let it settle.

    Returns (home joint vector, true object position).
    """
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()  

    q_home = data.qpos[:ARM_DOF].copy()
    adr = model.jnt_qposadr[free_joint_id]

    if REALLY_HARD_MODE:
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*HARD_BLIND_Y)
    elif HARD_SPAWN_MODE:
        x, y = HARD_SPAWNS[int(rng.integers(len(HARD_SPAWNS)))]
    elif RANDOMIZE_SPAWN:
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*SPAWN_Y_RANGE)
    else:
        x, y = TABLE_CENTRE

    data.qpos[adr:adr + 3] = [x, y, TABLE_HEIGHT + SPAWN_CLEARANCE]
    if RANDOMIZE_YAW:
        # cube: 90 deg symmetry, so [0, pi/2) covers every distinguishable yaw
        yaw = rng.uniform(0.0, np.pi / 2.0)
        data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    else:
        data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    q_home = data.qpos[:ARM_DOF].copy()
    for i in range(2000):
        data.ctrl[:ARM_DOF] = q_home
        data.ctrl[model.nu - 1] = GRIP_OPEN
        mujoco.mj_step(model, data)
        # the object only moves for the first ~200 steps; sync there so the fall
        # is visible, then step silently. A flat i % 5 here is 400 renders per
        # attempt, 390 of them showing an object already at rest.
        if viewer is not None:
            if i < 200 and i % SYNC_STEPS  == 0:
                viewer.sync()
            elif i >= 200 and i % 200 == 0:
                viewer.sync()

    return q_home, np.array(data.xpos[body_id]).copy()


def _detect_once(model, data, arm, camera, cam_id, rng, result, home_id,
                 free_joint_id, body_id, viewer, aborted, respawn=True):
    """Place the object once, then search for it.

    Two searches, in order of what actually clears the obstruction:

    1. Translate. The wrist camera sits 50 mm behind gripper/base, so anything
       more than ~50 mm ahead of the jaws is in the camera's own shadow, and no
       wrist rotation can move the camera past it. Moving the gripper forward is
       the only fix.
    2. Revolve. For close-range jaw occlusion that translation alone does not
       clear -- the camera looks past the jaws and the gap depends on yaw.

    Only the nominal view and the searched views are attempted; a miss returns
    None and the caller reports it. Nothing resets the scene.

    Returns (Detection, q_plan, truth), Detection None if nothing was found.
    """
    def stopped():
        return aborted is not None and aborted()

    def reached(q, target):
        """True if IK actually put gripper/base at target."""
        check = mujoco.MjData(model)
        check.qpos[:] = q
        mujoco.mj_forward(model, check)
        return np.linalg.norm(np.array(check.xpos[arm.base_id])
                              - np.array(target)) < 1e-3

    if respawn:
        q_home, truth = place_object(model, data, rng, home_id,
                                     free_joint_id, body_id, viewer=viewer)
    else:


        print(f'[DBG] no-respawn branch, object at '
              f'{np.round(data.xpos[body_id][:2], 3)}')        
        # keep the scene, but return to the nominal viewing pose. The search
        # offsets are relative to the current gripper position, so from the
        # post-place retreat pose they sweep the wrong part of the table.
        # Moving the arm does not move the object, so this is not a reset.
        truth = np.array(data.xpos[body_id]).copy()
        d_ref = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, d_ref, home_id)
        q_home = np.array(d_ref.qpos[:ARM_DOF])
        arm.move(q_home, GRIP_OPEN, rate=SEARCH_SPEED, viewer=viewer)
    if stopped():
        result.aborted = True
        return None, None, truth

    det = perceive(model, data, camera, cam_id)
    if det.ok and det.n_points >= MIN_OBJECT_POINTS:
        return det, data.qpos[:ARM_DOF].copy(), truth
    _log(f'[PERCEPTION] nominal view weak ({det.n_points} pts); searching')

    # the nominal pose is the anchor: offsets are applied to it, not accumulated
    anchor = np.array(data.xpos[arm.base_id], float)
    tgt_z = float(anchor[2])

    # --- 1. translate: get the camera ahead of the object -------------------
    for dx, dy in SEARCH_OFFSETS:
        target = [anchor[0] + dx, anchor[1] + dy, tgt_z]
        q = arm.solve_ik(target, data.qpos[:ARM_DOF], R_target=R_DOWN)
        if not reached(q, target):
            _log(f'[PERCEPTION] search ({dx:+.2f},{dy:+.2f}): IK failed')
            continue

        arm.move(q, GRIP_OPEN, rate=SEARCH_SPEED, viewer=viewer)
        result.n_peeks += 1
        if stopped():
            result.aborted = True
            return None, None, truth

        det = perceive(model, data, camera, cam_id)
        _log(f'[PERCEPTION] search ({dx:+.2f},{dy:+.2f}) -> {det.n_points} pts')
        if det.ok and det.n_points >= MIN_OBJECT_POINTS:
            return det, data.qpos[:ARM_DOF].copy(), truth

    # --- 2. revolve: change which gap the camera looks through --------------
    for ang in REVOLVE_DEG:
        R_ang = rotation_with_yaw(ang, R_DOWN)
        q = arm.solve_ik([anchor[0], anchor[1], tgt_z], data.qpos[:ARM_DOF],
                         R_target=R_ang)
        if not reached(q, [anchor[0], anchor[1], tgt_z]):
            _log(f'[PERCEPTION] revolve {ang:3d} deg: IK failed')
            continue

        arm.move(q, GRIP_OPEN, rate=REVOLVE_SPEED, viewer=viewer)
        result.n_peeks += 1
        if stopped():
            result.aborted = True
            return None, None, truth

        det = perceive(model, data, camera, cam_id)
        _log(f'[PERCEPTION] revolve {ang:3d} deg -> {det.n_points} pts')
        if det.ok and det.n_points >= MIN_OBJECT_POINTS:
            return det, data.qpos[:ARM_DOF].copy(), truth

    return None, None, truth


def find_object(model, data, arm, camera, cam_id, rng, result,
                home_id, free_joint_id, body_id, viewer=None, aborted=None,
                respawn=True):
    """Place the object once, then search for it.

    No resampling: a rejected view is recovered by revolving the wrist, which
    addresses the actual cause (occlusion) without discarding the scene. Earlier
    versions re-placed the object on a miss, which reset the whole scene every
    2000 steps and looked like the object teleporting around the table.

    Returns (Detection, q_plan, truth). Detection is None if nothing was found.
    """
    det, q_plan, truth = _detect_once(
        model, data, arm, camera, cam_id, rng, result, home_id,
        free_joint_id, body_id, viewer, aborted, respawn=respawn)

    return det, q_plan, truth

# ------------------------------------------------------------------ task ---
def pick_and_place(model, data, arm, camera, cam_id, rng,
                   viewer=None, aborted=None, respawn=True):
    """One pick-and-place attempt.

    respawn=True  reset the scene and place the object at a new random pose
    respawn=False keep the current scene and pick the object up where it lies
    """
    result = Result()

    def stopped():
        return aborted is not None and aborted()

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')

    det, q_plan, truth = find_object(
        model, data, arm, camera, cam_id, rng, result,
        home_id, free_joint_id, body_id, viewer=viewer, aborted=aborted,
        respawn=respawn)

    if det is None:
        print(f'[WARN] object not found after {result.n_samples} samples and '
              f'{result.n_peeks} peeks')
        return result

    result.found = True
    centre, yaw = det.centre, det.yaw
    result.yaw = yaw
    if truth is not None:
        result.perr = float(np.linalg.norm(centre - truth))

    _log(f'[PERCEPTION] {det.n_points} points  centre={np.round(centre, 4)}  '
         f'yaw={yaw:.1f} deg  error={result.perr:.4f} m')

    R_grasp = rotation_with_yaw(yaw, R_DOWN)
    R_place = R_grasp

    grasp_z = centre[2] + TIP_OFFSET
    q_approach = arm.solve_ik([centre[0], centre[1], grasp_z + APPROACH_Z],
                              q_plan, R_target=R_grasp)
    q_final = arm.solve_ik([centre[0], centre[1], grasp_z + FINAL_Z],
                           q_approach, R_target=R_grasp)
    q_grasp = arm.solve_ik([centre[0], centre[1], grasp_z],
                           q_final, R_target=R_grasp)
    q_lift = arm.solve_ik([centre[0], centre[1], grasp_z + LIFT_Z],
                          q_grasp, R_target=R_grasp)
    _log(f'[PLAN] grasp base z={grasp_z:.4f}')

    arm.move(q_approach, GRIP_OPEN, rate=RAPID, viewer=viewer)
    if stopped():
        result.aborted = True
        return result
    arm.move(q_final, GRIP_OPEN, rate=FEED, viewer=viewer)
    arm.move(q_grasp, GRIP_OPEN, rate=FEED, viewer=viewer)
    if stopped():
        result.aborted = True
        return result
    _log(f'[EXEC] at grasp base={np.round(data.xpos[arm.base_id], 4)}')

    arm.close_gripper(STEPS_CLOSE, viewer)
    if stopped():
        result.aborted = True
        return result

    arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=viewer)
    lifted = float(data.xpos[body_id][2]) - TABLE_HEIGHT
    result.picked = lifted > 0.06
    _log(f'[RESULT] PICK {"OK" if result.picked else "FAILED"} '
         f'-- object {lifted:+.4f} m above the surface')

    # ---- place ---------------------------------------------------------
    place_xy = (TABLE_CENTRE[0] + PLACE_DX + PLACE_OFFSET_X, TABLE_CENTRE[1] + PLACE_DY + PLACE_OFFSET_Y)
    place_base_z = grasp_z + PLACE_CLEARANCE

    q_place_app = arm.solve_ik([place_xy[0], place_xy[1],
                                place_base_z + APPROACH_Z],
                               q_lift, R_target=R_place)
    q_place = arm.solve_ik([place_xy[0], place_xy[1], place_base_z],
                           q_place_app, R_target=R_place)
    q_retreat = arm.solve_ik([place_xy[0], place_xy[1],
                              place_base_z + LIFT_Z],
                             q_place, R_target=R_place)

    arm.move(q_place_app, GRIP_CLOSED, rate=RAPID, viewer=viewer)
    if stopped():
        result.aborted = True
        return result
    arm.move(q_place, GRIP_CLOSED, rate=FEED, viewer=viewer)
    if stopped():
        result.aborted = True
        return result

    arm.open_gripper(STEPS_RELEASE, viewer)
    if stopped():
        result.aborted = True
        return result

    arm.move(q_retreat, GRIP_OPEN, rate=RAPID, viewer=viewer)

    placed = np.array(data.xpos[body_id])
    result.placed_err = float(np.linalg.norm(placed[:2] - np.array(place_xy)))
    result.placed = abs(placed[2] - (TABLE_HEIGHT + OBJECT_HEIGHT / 2.0)) < 0.005
    _log(f'[RESULT] PLACE {"OK" if result.placed else "FAILED"} '
         f'-- {result.placed_err * 1000:.1f} mm from target')

    return result