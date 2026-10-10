
# vision-guided pick with UR5e + 2F-85 + D435i.

# Perception finds the object, IK plans a top-down grasp, the arm executes it.
# Press R to re-run, close the viewer window to exit.

import numpy as np
import mujoco
import mujoco.viewer
import mink

from src.model_loader import (
    load_robot_model,
    TABLE_HEIGHT, TABLE_CENTRE, TABLE_HALF, OBJECT_SIZE, OBJECT_HEIGHT,
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

STEPS_PEEK = 1500        # wrist reposition for a second look

MOTION_SPEED = 1.0       # scale speed 

GRASP_HEIGHT_FRAC = 0.2    
TIP_OFFSET = 0.1094 - (GRASP_HEIGHT_FRAC - 0.5) * OBJECT_HEIGHT

R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])   # gripper +z aligned with world -z

ARM_DOF = 6

RANDOMIZE_SPAWN = True
SPAWN_X = (-0.30, 0.05)      # world x range for the object
SPAWN_Y = (0.40, 0.62)       # world y range
SPAWN_MARGIN = 0.06
SPAWN_Y_MIN = 0.45
SPAWN_CLEARANCE = 0.032

SPAWN_X_RANGE = (TABLE_CENTRE[0] - TABLE_HALF[0] + SPAWN_MARGIN + OBJECT_SIZE[0],
                 TABLE_CENTRE[0] + TABLE_HALF[0] - SPAWN_MARGIN - OBJECT_SIZE[0])
SPAWN_Y_RANGE = (max(TABLE_CENTRE[1] - TABLE_HALF[1] + SPAWN_MARGIN + OBJECT_SIZE[1], SPAWN_Y_MIN),
                 TABLE_CENTRE[1] + TABLE_HALF[1] - SPAWN_MARGIN - OBJECT_SIZE[1])

MIN_OBJECT_POINTS = 1500      # clean views give ~3000-5000; occluded ones <1500
PERCEPTION_TRIES = 5

VIEW_Y_LEAD = 0.05
VIEW_OFFSETS = ((0.0, 0.0), (0.0, VIEW_Y_LEAD), (0.0, -VIEW_Y_LEAD))


# Deliberately awkward spawns for exercising the perception-recovery path.
# (x, y) only; height comes from SPAWN_CLEARANCE. Measured against this scene:
HARD_SPAWNS = (
    # occluded by the gripper: ~540 pts, 27 mm xy error from a clipped bbox
    (-0.219, 0.411),
    # occluded by the gripper: ~1020 pts, 20 mm xy error
    (0.030, 0.400),
    # outside the home-pose camera footprint: 0 pts found
    (-0.200, 0.350),
    # marginal footprint, just inside the far edge
    (-0.320, 0.700),
)
HARD_SPAWN_MODE = True     # True -> ignore the random range, use HARD_SPAWNS


# Place target. 
PLACE_DX = 0.0          # offset from TABLE_CENTRE in x
PLACE_DY = 0.0          # offset from TABLE_CENTRE in y
STEPS_PLACE_APPROACH = 2000
STEPS_PLACE_DOWN = 1500
STEPS_RELEASE = 2000
STEPS_RETREAT = 1500

ARM_SPEED = 0.04        # m/s, at MOTION_SPEED = 1.0
MIN_PHASE_STEPS = 400    # so very short moves still ramp rather than snap

PLACE_CLEARANCE = 0.002      

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



def peek(model, data, cam_id, q_home, obj_xyz, dx, dy):
    """Move the wrist sideways so the fingers stop occluding the object.

    Offsets gripper/base in world x/y from the object's position, keeping the
    approach axis pointing down. Returns the joint vector actually reached, or
    q_home unchanged if every offset failed IK.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    base_z = float(data.xpos[base_id][2])
    obj_xyz = np.asarray(obj_xyz, float)

    # --- ADD 1: orientation to hold, taken from the pose we are peeking from ---
    d_hold = mujoco.MjData(model)
    d_hold.qpos[:] = data.qpos          # full nq; q_home is arm-only
    mujoco.mj_forward(model, d_hold)
    R_hold = np.array(d_hold.xmat[base_id]).reshape(3, 3)
    # --------------------------------------------------------------------------

    for scale in (1.0, 1.6, 2.4):
        target = [obj_xyz[0] + dx * scale, obj_xyz[1] + dy * scale, base_z]

        # --- CHANGE 2: pin the orientation so wrist_3 cannot spin ~1.77 rad ---
        q = solve_ik(model, target, q_home, R_target=R_hold)
        # ----------------------------------------------------------------------

        check = mujoco.MjData(model)
        check.qpos[:] = q
        mujoco.mj_forward(model, check)
        posed = np.array(check.xpos[base_id])
        if np.linalg.norm(posed - np.array(target)) < 5e-3:
            cam = np.array(check.cam_xpos[cam_id])
            if cam[2] > TABLE_HEIGHT + 0.05:
                return q
    return q_home

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

def solve_ik(model, target, q_init, R_target=R_DOWN, iters=600,
             gain_p=0.5, gain_o=0.5, damp=1e-2, max_joint=3.0):

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
        eR = R_target @ R.T
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
    free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'target_free')
    rng = np.random.default_rng()

    camera = RGBDCamera(model, cam_name, width=640, height=480)
    print(f'[INFO] intrinsics K:\n{camera.K}')

    restart = [False]

    def steps_for(q_from, q_to):
        """Step count that covers the joint-space move at ARM_SPEED."""
        return max(MIN_PHASE_STEPS,
                   int(np.linalg.norm(np.asarray(q_to)[:arm_dof]
                                      - np.asarray(q_from)[:arm_dof])
                       / ARM_SPEED / model.opt.timestep))

    def on_key(keycode):
        if keycode in (ord('R'), ord('r')):
            restart[0] = True

    def move_arm(q_target, viewer, grip=GRIP_OPEN, ease='quintic'):

        q_start = data.qpos[:ARM_DOF].copy()
        dist = float(np.linalg.norm(np.asarray(q_target)[:ARM_DOF]
                                    - np.asarray(q_start)[:ARM_DOF]))
        steps = max(MIN_PHASE_STEPS,
                    int(dist / ARM_SPEED / model.opt.timestep))
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

    def open_gripper(steps, viewer):
        """Ramp the fingers open while holding the present arm pose."""
        q_hold = data.qpos[:ARM_DOF].copy()
        for i in range(steps):
            a = (i + 1) / steps
            data.ctrl[:ARM_DOF] = q_hold
            data.ctrl[grip_act] = GRIP_CLOSED + a * (GRIP_OPEN - GRIP_CLOSED)
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

            # ---- reset scene, randomise object position and let it settle, move camera back if occluded  ------------------
            estimate = None
            q_plan = data.qpos[:ARM_DOF].copy()

            for attempt in range(PERCEPTION_TRIES):
                mujoco.mj_resetDataKeyframe(model, data, home_id)
                mujoco.mj_forward(model, data)

                if HARD_SPAWN_MODE:
                    hx, hy = HARD_SPAWNS[attempt % len(HARD_SPAWNS)]
                    adr = model.jnt_qposadr[free_joint_id]
                    data.qpos[adr:adr + 3] = [hx, hy, TABLE_HEIGHT + SPAWN_CLEARANCE]
                    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
                    mujoco.mj_forward(model, data)

                elif RANDOMIZE_SPAWN:
                    adr = model.jnt_qposadr[free_joint_id]
                    data.qpos[adr:adr + 3] = [rng.uniform(*SPAWN_X_RANGE),
                                              rng.uniform(*SPAWN_Y_RANGE),
                                              TABLE_HEIGHT + SPAWN_CLEARANCE]
                    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
                    mujoco.mj_forward(model, data)

                viewer.sync() 
                q_home = data.qpos[:ARM_DOF].copy()
                for _ in range(2000):
                    data.ctrl[:ARM_DOF] = q_home
                    data.ctrl[grip_act] = GRIP_OPEN
                    mujoco.mj_step(model, data)
                    if _ % 50 == 0:
                        viewer.sync()

                est, n_pts, _ = perceive(model, data, camera, cam_id)
                if est is None or n_pts < MIN_OBJECT_POINTS:
                    print(f'[PERCEPTION] nominal view weak ({n_pts} points); '
                          f'peeking around the gripper')
                    for dx, dy in VIEW_OFFSETS[1:]:
                        q_peek = peek(model, data, cam_id, q_home, est, dx, dy)
                        move_arm(q_peek, viewer)
                        if restart[0]:
                            break
                        est2, n2, _ = perceive(model, data, camera, cam_id)
                        print(f'[PERCEPTION] peek dx={dx:+.3f} dy={dy:+.3f} '
                              f'-> {n2} points')
                        if est2 is not None and n2 >= MIN_OBJECT_POINTS:
                            est, n_pts = est2, n2
                            q_plan = data.qpos[:ARM_DOF].copy()
                            break
                        move_arm(q_home, viewer)   # back to nominal
                        if restart[0]:
                            break
                        # data.qpos[:ARM_DOF] = q_peek[:ARM_DOF]
                        # mujoco.mj_forward(model, data)
                        # for _ in range(300):
                        #     data.ctrl[:ARM_DOF] = q_peek[:ARM_DOF]
                        #     data.ctrl[grip_act] = GRIP_OPEN
                        #     mujoco.mj_step(model, data)
                        est2, n2, _ = perceive(model, data, camera, cam_id)
                        print(f'[PERCEPTION] peek dx={dx:+.3f} dy={dy:+.3f} '
                              f'-> {n2} points')
                        if est2 is not None and n2 >= MIN_OBJECT_POINTS:
                            est, n_pts = est2, n2
                            q_plan = data.qpos[:ARM_DOF].copy()
                            break

                if est is not None and n_pts >= MIN_OBJECT_POINTS:
                    estimate = est
                    q_plan = data.qpos[:ARM_DOF].copy()
                    break
                print(f'[PERCEPTION] rejected sample ({n_pts} points); resampling')

            if estimate is None:
                print('[WARN] no usable perception after '
                      f'{PERCEPTION_TRIES} attempts; press R to retry')
                while viewer.is_running() and not restart[0]:
                    viewer.sync()
                continue

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
                                          grasp_z + APPROACH_Z], q_plan)
            q_final = solve_ik(model, [estimate[0], estimate[1],
                                       grasp_z + FINAL_Z], q_approach)
            q_grasp = solve_ik(model, [estimate[0], estimate[1], grasp_z], q_final)
            q_lift = solve_ik(model, [estimate[0], estimate[1],
                                      grasp_z + LIFT_Z], q_grasp)

            print(f'[PLAN] grasp base z={grasp_z:.4f}, lift base z={grasp_z + LIFT_Z:.4f}')


            move_arm(q_approach, viewer, grip=GRIP_OPEN)
            if restart[0]:
                continue
            print(f'[EXEC] approach  base={np.round(data.xpos[base_id], 4)}')

            move_arm(q_final, viewer, grip=GRIP_OPEN)   # descend, still open
            move_arm(q_grasp, viewer, grip=GRIP_OPEN)
            if restart[0]:
                continue
            print(f'[EXEC] at grasp  base={np.round(data.xpos[base_id], 4)} '
                  f'obj={np.round(data.xpos[body_id], 4)}')

            close_gripper(GRIP_CLOSED, scaled(STEPS_CLOSE), viewer)        # close in place
            if restart[0]: 
                continue
            print(f'[EXEC] closed    obj={np.round(data.xpos[body_id], 4)}')

            move_arm(q_lift, viewer, grip=GRIP_CLOSED)   # lift, holding

            axis = np.array(data.xmat[base_id]).reshape(3, 3)[:, 2]
            print(f'[EXEC] grasp axis={np.round(axis, 4)} '
                  f'tilt={np.degrees(np.arccos(np.clip(-axis[2], -1, 1))):.1f} deg')

            obj = np.array(data.xpos[body_id])
            height = obj[2] - TABLE_HEIGHT
            ok = height > 0.06
            print(f'[RESULT] {"PICK SUCCEEDED" if ok else "PICK FAILED"} '
                  f'-- object {height:+.4f} m above the surface')

            # ---- place --------------------------------------------------
            # The fingers must reach the surface exactly, so the drop height is
            # object-centre height + the base->fingertip offset.
            place_xy = (TABLE_CENTRE[0] + PLACE_DX, TABLE_CENTRE[1] + PLACE_DY)
            #place_base_z = TABLE_HEIGHT + OBJECT_HEIGHT / 2.0 + TIP_OFFSET
            place_base_z = grasp_z + PLACE_CLEARANCE

            q_place_app = solve_ik(model, [place_xy[0], place_xy[1],
                                           place_base_z + APPROACH_Z], q_lift)
            q_place = solve_ik(model, [place_xy[0], place_xy[1],
                                       place_base_z], q_place_app)
            q_retreat = solve_ik(model, [place_xy[0], place_xy[1],
                                         place_base_z + LIFT_Z], q_place)

            print(f'[PLAN] place at ({place_xy[0]:+.3f}, {place_xy[1]:+.3f}) '
                  f'base z={place_base_z:.4f}')

            move_arm(q_place_app, viewer, grip=GRIP_CLOSED)
            if restart[0]:
                continue
            move_arm(q_place, viewer, grip=GRIP_CLOSED)
            if restart[0]:
                continue

            print(f'[EXEC] placed {np.round(data.xpos[base_id], 4)} '
                  f'obj={np.round(data.xpos[body_id], 4)}')

            open_gripper(scaled(STEPS_RELEASE), viewer)
            if restart[0]:
                continue

            move_arm(q_retreat, viewer, grip=GRIP_OPEN)

            placed = np.array(data.xpos[body_id])
            err = np.linalg.norm(placed[:2] - np.array(place_xy))
            resting = abs(placed[2] - (TABLE_HEIGHT + OBJECT_HEIGHT / 2.0)) < 0.005
            print(f'[RESULT] PLACE {"OK" if resting else "FAILED"} -- '
                  f'object at {np.round(placed, 4)}, '
                  f'{err * 1000:.1f} mm from target')

            # ---- hold until R or window close ---------------------------
            print('[INFO] holding -- R to re-run, close the window to exit')
            while viewer.is_running() and not restart[0]:
                data.ctrl[:ARM_DOF] = data.qpos[:ARM_DOF]   # stiff position hold
                data.ctrl[grip_act] = GRIP_CLOSED
                mujoco.mj_step(model, data)
                viewer.sync()

if __name__ == '__main__':
    main()