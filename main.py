
# vision-guided pick with UR5e + 2F-85 + D435i.

# Perception finds the object, IK plans a top-down grasp, the arm executes it.
# Press R to re-run, close the viewer window to exit.

import numpy as np
import mujoco
import mujoco.viewer
import mink

from src.model_loader import (
    load_robot_model,
    TABLE_HEIGHT, OBJECT_HEIGHT,
)
from src.camera import RGBDCamera

# Constants
SELF_FILTER = 0.20       # drop everything nearer the lens than this (the gripper)
SURFACE_BAND = 0.005     # points this far above the work surface are kept
DEPTH_MAX = 5.0          # metres; MuJoCo returns ~100.003 for pixels that hit nothing
GRIP_OPEN = 0.0          # fingers_actuator command
GRIP_CLOSED = 255.0
APPROACH_Z = 0.10        # how far above the object to stage the approach
LIFT_Z = 0.15            # how far to raise after grasping

FINAL_Z = 0.03           # height at which the slow crawl begins
STEPS_APPROACH = 2000    # fast, from home to the staging pose
STEPS_FINAL = 1500       # slow crawl, staging -> just above the object
STEPS_GRASP = 1200       # final contact descent
STEPS_CLOSE = 3000
STEPS_LIFT = 3000

MOTION_SPEED = 1.0       # scale speed 

GRASP_HEIGHT_FRAC = 0.2    
TIP_OFFSET = 0.12 - (GRASP_HEIGHT_FRAC - 0.5) * OBJECT_HEIGHT

R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])   # gripper +z aligned with world -z

ARM_DOF = 6

def find_camera(model):
    """Return (name, id) of the wrist camera."""
    for candidate in ('camera/depth_cam', 'depth_cam'):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
        if cid >= 0:
            return candidate, cid
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and 'depth_cam' in name:
            return name, i
    return None, -1


def perceive(model, data, camera, cam_id):
    """Locate the object on the work surface.

    The floor plane is *not* fitted by RANSAC here: with the object close to a
    downward camera it has more inliers than the surface, and the fit locks onto
    the object's top face. The work surface height is known, so use it.
    """
   #DEPTH_MAX = 5.0          # metres; MuJoCo returns ~100 for empty pixels

    rgb, depth = camera.capture(data)
    points = camera.deproject(depth, data)

    cam_pos = data.cam_xpos[cam_id]
    finite = np.isfinite(points).all(axis=-1)
    near = np.linalg.norm(points - cam_pos, axis=-1)

    keep = (finite
            & (near > SELF_FILTER)
            & (near < DEPTH_MAX)
            & (points[..., 2] > TABLE_HEIGHT + SURFACE_BAND))

    obj = points[keep]
    if len(obj) < 50:
        return None, len(obj), None

    centre_xy = 0.5 * (obj[:, :2].min(axis=0) + obj[:, :2].max(axis=0))
    # A downward camera only ever sees the top face, so the centroid's z is the
    # top of the object, not its centre. Derive the centre from the known
    # surface height plus half the known object height.
    centre_z = TABLE_HEIGHT + OBJECT_HEIGHT / 2.0
    estimate = np.array([centre_xy[0], centre_xy[1], centre_z])

    # perpendicular distance from the surface plane, per surviving point
    return estimate, len(obj), obj

def scaled(steps):
    return max(1, int(round(steps / MOTION_SPEED)))

def solve_ik(model, target, q_init, iters=600, gain_p=0.5, gain_o=0.5,
             damp=1e-2, max_joint=3.0):
    """DLS IK for gripper/base -- position AND orientation -- seeded near q_init.

    Position-only IK reaches the right point with the tool tilted (30 deg at the
    grasp pose), leaving the fingers above the object and closing along a skewed
    axis. The orientation term pins the approach axis to world -z via R_DOWN.

    Operates on the arm joints only; the object freejoint keeps its keyframe value.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    target = np.asarray(target, float)

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d,
                                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home'))

    q_arm = np.array(q_init[:ARM_DOF], float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(iters):
        d.qpos[:ARM_DOF] = q_arm
        mujoco.mj_forward(model, d)

        err_p = target - np.array(d.xpos[base_id])

        R = np.array(d.xmat[base_id]).reshape(3, 3)
        eR = R_DOWN @ R.T
        err_o = np.array([eR[2, 1] - eR[1, 2],
                          eR[0, 2] - eR[2, 0],
                          eR[1, 0] - eR[0, 1]]) * 0.5

        if np.linalg.norm(err_p) < 1e-5 and np.linalg.norm(err_o) < 1e-6:
            break

        mujoco.mj_jacBody(model, d, jacp, jacr, base_id)
        J = np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])
        e = np.concatenate([gain_p * err_p, gain_o * err_o])
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        q_arm = np.clip(q_arm + dq, -max_joint, max_joint)

    q_full = d.qpos.copy()
    q_full[:ARM_DOF] = q_arm
    return q_full

def verify_ik(model):
    """Self-test: drive IK to a known target and report position + orientation."""
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, home)
    mujoco.mj_forward(model, d)
    q0 = d.qpos.copy()

    target = np.array([-0.30, 0.45, 0.30])
    q = solve_ik(model, target, q0)

    d2 = mujoco.MjData(model)
    d2.qpos[:] = q
    mujoco.mj_forward(model, d2)
    pos_err = np.linalg.norm(np.array(d2.xpos[base_id]) - target)
    axis = np.array(d2.xmat[base_id]).reshape(3, 3)[:, 2]
    tilt = np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))
    print(f'[IK-CHECK] pos_err={pos_err:.5f} m  tilt={tilt:.2f} deg  axis={np.round(axis, 4)}')
    print(f'[IK-CHECK] expected pos_err<0.001 and tilt<1.0; '
        f'q={np.round(q[:6], 4)}')
    return tilt

def main():
    model = load_robot_model()
    if model is None:
        return

    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_id < 0:
        print('[ERROR] model has no "home" keyframe; cannot start.')
        return
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    cam_name, cam_id = find_camera(model)
    if cam_id < 0:
        print('[ERROR] no wrist camera found; cannot proceed.')
        return
    print(f'[INFO] {model.nbody} bodies, {model.njnt} joints, '
          f'{model.nu} actuators; wrist camera = {cam_name}')

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    grip_act = model.nu - 1                       # fingers_actuator

    camera = RGBDCamera(model, cam_name, width=640, height=480)
    print(f'[INFO] intrinsics K:\n{camera.K}')

    restart = [False]

    def on_key(keycode):
        if keycode in (ord('R'), ord('r')):
            restart[0] = True

    def move_arm(q_target, steps, viewer, grip=GRIP_OPEN, ease='quintic'):
        q_start = data.qpos[:ARM_DOF].copy()
        for i in range(steps):
            a = (i + 1) / steps
            if ease == 'quintic':
                a = a * a * a * (a * (a * 6.0 - 15.0) + 10.0)
            else:
                a = a * a * (3.0 - 2.0 * a)
            data.ctrl[:ARM_DOF] = a * q_target[:ARM_DOF] + (1 - a) * q_start
            data.ctrl[grip_act] = grip
            mujoco.mj_step(model, data)
            if viewer is not None and i % 25 == 0:
                viewer.sync()


    def close_gripper(grip_end, steps, viewer):
        """Ramp the fingers shut while holding the present arm pose.

        The grip command is interpolated rather than stepped: commanding
        0 -> 255 in one tick drives the fingers into the object and the contact
        solver spikes to ~412 N. Ramping over this phase gives the same final
        grip at ~80 N peak.
        """
        q_hold = data.qpos[:ARM_DOF].copy()
        for i in range(steps):
            a = (i + 1) / steps
            data.ctrl[:ARM_DOF] = q_hold
            data.ctrl[grip_act] = GRIP_OPEN + a * (grip_end - GRIP_OPEN)
            mujoco.mj_step(model, data)
            if viewer is not None and i % 25 == 0:
                viewer.sync()

    verify_ik(model)
    print(f"[INFO] GRAVITY COMPENSATION: {model.body_gravcomp}")          # per-body; 0 for off
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        while viewer.is_running():
            restart[0] = False

            # ---- reset scene and let it settle -------------------------
            mujoco.mj_resetDataKeyframe(model, data, home_id)
            mujoco.mj_forward(model, data)
            q_home = data.qpos[:ARM_DOF].copy()
            for _ in range(2000):
                data.ctrl[:ARM_DOF] = q_home          # hold the home pose
                data.ctrl[grip_act] = GRIP_OPEN
                mujoco.mj_step(model, data)

            # ---- perceive ----------------------------------------------
            estimate, n_pts, _ = perceive(model, data, camera, cam_id)
            if estimate is None:
                print(f'[WARN] object not found ({n_pts} candidate points); '
                      f'press R to retry')
                while viewer.is_running() and not restart[0]:
                    viewer.sync()
                continue

            truth = np.array(data.xpos[body_id])
            print(f'[PERCEPTION] {n_pts} object points  estimate={np.round(estimate, 4)}'
                  f'  truth={np.round(truth, 4)}  error={np.linalg.norm(estimate - truth):.4f} m')

            # ---- plan ---------------------------------------------------
            grasp_z = estimate[2] + TIP_OFFSET
            q_approach = solve_ik(model, [estimate[0], estimate[1],
                                          grasp_z + APPROACH_Z], q_home)
            q_final = solve_ik(model, [estimate[0], estimate[1],
                                       grasp_z + FINAL_Z], q_approach)
            q_grasp = solve_ik(model, [estimate[0], estimate[1], grasp_z], q_final)
            q_lift = solve_ik(model, [estimate[0], estimate[1],
                                      grasp_z + LIFT_Z], q_grasp)

            print(f'[PLAN] grasp base z={grasp_z:.4f}, lift base z={grasp_z + LIFT_Z:.4f}')


            move_arm(q_approach, scaled(STEPS_APPROACH), viewer, grip=GRIP_OPEN)
            if restart[0]:
                continue
            print(f'[EXEC] approach  base={np.round(data.xpos[base_id], 4)}')

            move_arm(q_final,    scaled(STEPS_FINAL),    viewer, grip=GRIP_OPEN)   # descend, still open
            move_arm(q_grasp,    scaled(STEPS_GRASP),    viewer, grip=GRIP_OPEN)
            if restart[0]:
                continue
            print(f'[EXEC] at grasp  base={np.round(data.xpos[base_id], 4)} '
                  f'obj={np.round(data.xpos[body_id], 4)}')

            close_gripper(GRIP_CLOSED, scaled(STEPS_CLOSE), viewer)         # close in place
            if restart[0]: 
                continue
            print(f'[EXEC] closed    obj={np.round(data.xpos[body_id], 4)}')

            move_arm(q_lift,     scaled(STEPS_LIFT),     viewer, grip=GRIP_CLOSED)   # lift, holding

            axis = np.array(data.xmat[base_id]).reshape(3, 3)[:, 2]
            print(f'[EXEC] grasp axis={np.round(axis, 4)} '
                  f'tilt={np.degrees(np.arccos(np.clip(-axis[2], -1, 1))):.1f} deg')

            obj = np.array(data.xpos[body_id])
            height = obj[2] - TABLE_HEIGHT
            ok = height > 0.06
            print(f'[RESULT] {"PICK SUCCEEDED" if ok else "PICK FAILED"} '
                  f'-- object {height:+.4f} m above the surface')

            # ---- hold until R or window close ---------------------------
            print('[INFO] holding -- R to re-run, close the window to exit')
            while viewer.is_running() and not restart[0]:
                data.ctrl[:ARM_DOF] = data.qpos[:ARM_DOF]   # stiff position hold
                data.ctrl[grip_act] = GRIP_CLOSED
                mujoco.mj_step(model, data)
                viewer.sync()

if __name__ == '__main__':
    main()