"""One pick-and-place attempt, end to end.

Owns scene reset and object placement, the perception retry policy, and the
grasp/place sequence. Knows nothing about the viewer loop or restart handling
beyond a callback that lets it bail out mid-sequence.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, RAPID, FEED,
    TABLE_HEIGHT, TABLE_CENTRE, OBJECT_HEIGHT,
    SPAWN_X_RANGE, SPAWN_Y_RANGE, SPAWN_CLEARANCE, 
    HARD_SPAWNS, HARD_SPAWN_MODE, REALLY_HARD_MODE, HARD_BLIND_Y,
    RANDOMIZE_SPAWN, RANDOMIZE_YAW,
    MIN_OBJECT_POINTS, PERCEPTION_TRIES, VIEW_OFFSETS,
    TIP_OFFSET, APPROACH_Z, FINAL_Z, LIFT_Z,
    PLACE_DX, PLACE_DY, PLACE_CLEARANCE, STEPS_CLOSE, STEPS_RELEASE,
)
from src.perception import perceive, rotation_with_yaw


@dataclass
class Result:
    picked: bool = False
    placed: bool = False
    yaw: float = 0.0
    perr: float = float('nan')     # perception error vs truth (sim only)
    placed_err: float = float('nan')
    aborted: bool = False          # caller asked to restart mid-sequence


# scene setup
def place_object(model, data, rng, home_id, free_joint_id):
    """Reset the scene, put the object somewhere, and let it settle.

    Returns the home joint vector and the object's true position.
    """
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    adr = model.jnt_qposadr[free_joint_id]

    if REALLY_HARD_MODE:
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*HARD_BLIND_Y)
    elif HARD_SPAWN_MODE:
        x, y = HARD_SPAWNS[rng.integers(len(HARD_SPAWNS))]
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
    for _ in range(2000):
        data.ctrl[:ARM_DOF] = q_home
        data.ctrl[model.nu - 1] = GRIP_OPEN
        mujoco.mj_step(model, data)

    return q_home, np.array(data.xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, 'target')]).copy()


# perception 
def find_object(arm, camera, cam_id, model, data, rng, home_id,
                free_joint_id, viewer=None, aborted=None):
    """Place the object and locate it, resampling then peeking if occluded.

    Resample first: it costs one reset plus one settle, and most occluded samples
    clear on a new draw. Moving the camera costs arm moves, so it is the last
    resort.

    Returns (Detection, q_plan, truth) or (None, None, truth).
    """
    det = None
    estimate = None
    q_home = None
    truth = None

    for _ in range(PERCEPTION_TRIES):
        q_home, truth = place_object(model, data, rng, home_id, free_joint_id)
        if viewer is not None:
            viewer.sync()

        det = perceive(model, data, camera, cam_id)
        if det.ok and det.n_points >= MIN_OBJECT_POINTS:
            break
        print(f'[PERCEPTION] rejected sample ({det.n_points} points); resampling')

    if det is not None and det.ok and det.n_points >= MIN_OBJECT_POINTS:
        return det, data.qpos[:ARM_DOF].copy(), truth

    # every draw was occluded (or nothing was in view) -- try moving the camera
    if q_home is not None:
        print('[PERCEPTION] no clean sample; peeking around the gripper')
        for dx, dy in VIEW_OFFSETS[1:]:
            anchor = (det.centre if (det is not None and det.ok)
                      else np.array(data.xpos[free_joint_id and mujoco.mj_name2id(
                          model, mujoco.mjtObj.mjOBJ_BODY, 'target')]))
            q_peek = arm.peek_target(q_home, anchor, dx, dy)
            arm.move(q_peek, GRIP_OPEN, viewer=viewer)
            if aborted is not None and aborted():
                return None, None, truth

            det2 = perceive(model, data, camera, cam_id)
            print(f'[PERCEPTION] peek dx={dx:+.3f} dy={dy:+.3f} '
                  f'-> {det2.n_points} points')
            if det2.ok and det2.n_points >= MIN_OBJECT_POINTS:
                return det2, data.qpos[:ARM_DOF].copy(), truth

            arm.move(q_home, GRIP_OPEN, viewer=viewer)
            if aborted is not None and aborted():
                return None, None, truth

    return None, None, truth


# task 
def pick_and_place(model, data, arm, camera, cam_id, rng, viewer=None,
                   aborted=None):
    """One pick-and-place attempt. Never raises for a perception miss."""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    result = Result()

    def stopped():
        return aborted is not None and aborted()

    det, q_plan, truth = find_object(arm, camera, cam_id, model, data, rng,
                                     home_id, free_joint_id,
                                     viewer=viewer, aborted=aborted)
    if det is None:
        print(f'[WARN] object not found after {PERCEPTION_TRIES} samples and '
              f'{len(VIEW_OFFSETS) - 1} peeks; press R to retry')
        return result

    estimate, yaw = det.centre, det.yaw
    result.yaw = yaw
    if truth is not None:
        result.perr = float(np.linalg.norm(estimate - truth))
    print(f'[PERCEPTION] {det.n_points} points  centre={np.round(estimate, 4)}  '
          f'yaw={yaw:.1f} deg  truth={np.round(truth, 4)}  '
          f'error={result.perr:.4f} m')

    R_grasp = rotation_with_yaw(yaw, np.array([[1.0, 0.0, 0.0],
                                               [0.0, -1.0, 0.0],
                                               [0.0, 0.0, -1.0]]))
    R_place = R_grasp

    grasp_z = estimate[2] + TIP_OFFSET
    q_approach = arm.solve_ik([estimate[0], estimate[1], grasp_z + APPROACH_Z],
                              q_plan, R_target=R_grasp)
    q_final = arm.solve_ik([estimate[0], estimate[1], grasp_z + FINAL_Z],
                           q_approach, R_target=R_grasp)
    q_grasp = arm.solve_ik([estimate[0], estimate[1], grasp_z],
                           q_final, R_target=R_grasp)
    q_lift = arm.solve_ik([estimate[0], estimate[1], grasp_z + LIFT_Z],
                          q_grasp, R_target=R_grasp)
    print(f'[PLAN] grasp base z={grasp_z:.4f}, lift base z={grasp_z + LIFT_Z:.4f}')

    arm.move(q_approach, GRIP_OPEN, rate=RAPID, viewer=viewer)
    if stopped():
        result.aborted = True
        return result
    arm.move(q_final, GRIP_OPEN, rate=FEED, viewer=viewer)
    arm.move(q_grasp, GRIP_OPEN, rate=FEED, viewer=viewer)
    if stopped():
        result.aborted = True
        return result
    print(f'[EXEC] at grasp  base={np.round(data.xpos[arm.base_id], 4)} '
          f'obj={np.round(data.xpos[
              mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")], 4)}')

    arm.close_gripper(STEPS_CLOSE, viewer)
    if stopped():
        result.aborted = True
        return result

    arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=viewer)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    lifted = float(data.xpos[body_id][2]) - TABLE_HEIGHT
    result.picked = lifted > 0.06
    print(f'[RESULT] {"PICK SUCCEEDED" if result.picked else "PICK FAILED"} '
          f'-- object {lifted:+.4f} m above the surface')

    # place 
    place_xy = (TABLE_CENTRE[0] + PLACE_DX, TABLE_CENTRE[1] + PLACE_DY)
    place_base_z = grasp_z + PLACE_CLEARANCE

    q_place_app = arm.solve_ik([place_xy[0], place_xy[1],
                                place_base_z + APPROACH_Z],
                               q_lift, R_target=R_place)
    q_place = arm.solve_ik([place_xy[0], place_xy[1], place_base_z],
                           q_place_app, R_target=R_place)
    q_retreat = arm.solve_ik([place_xy[0], place_xy[1],
                              place_base_z + LIFT_Z],
                             q_place, R_target=R_place)
    print(f'[PLAN] place at ({place_xy[0]:+.3f}, {place_xy[1]:+.3f}) '
          f'base z={place_base_z:.4f}')

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
    print(f'[RESULT] PLACE {"OK" if result.placed else "FAILED"} -- '
          f'object at {np.round(placed, 4)}, '
          f'{result.placed_err * 1000:.1f} mm from target')

    return result