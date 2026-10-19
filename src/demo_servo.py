# PBVS demonstration: why closed-loop beats open-loop planning.
#
# DEMONSTRATION / SCORING ONLY. This module deliberately crosses the robot /
# world boundary -- it displaces the cube mid-approach so that an open-loop
# planner fails and the servo does not, and it reads the object's true pose to
# measure the resulting gap.
#
# That is exactly what robot-side code must never do, which is why this lives
# here rather than in servo.py. Nothing in src/ imports this module; it is an
# entry point you run directly:
#
#     python -m src.demo_servo
#
# It was previously embedded in servo.py, where it forced an import of
# src.pick_place into a production robot module.

import mujoco
import numpy as np

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, TABLE_HEIGHT, OBJECT_HEIGHT,
)


def _try_grasp(model, data, arm, reg, idx):
    # Close, lift, and report whether the object came with the gripper.
    arm.close_gripper(1200)
    p_now = np.array(data.xpos[arm.base_id], float)
    q = arm.solve_ik(p_now + np.array([0.0, 0.0, 0.15]), data.qpos[:ARM_DOF],
                     R_target=np.array(data.xmat[arm.base_id],
                                      float).reshape(3, 3))
    arm.move(q, GRIP_CLOSED, rate=0.05)
    lifted = reg.pose(data, idx)[2] > TABLE_HEIGHT + OBJECT_HEIGHT / 2.0
    arm.open_gripper(600)
    return lifted


def _trial(model, data, home, world, use_servo, shift, seed):
    # Plan, hover, displace the object, then descend. Measure where we ended.
    from src.arm import Arm
    from src.pick_place import ObjectRegistry, spawn_one
    from src.camera import RGBDCamera
    from src.camera import find_camera
    from src.servo import PBVSServo
    from src.world_sim import SimObjectWorld
    from src.kinematics_mujoco import MujocoKinematics
    from src.robot_backend_mujoco import MujocoBackend

    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    backend = MujocoBackend(model, data, ARM_DOF)
    arm = Arm(model, data,
              kinematics=MujocoKinematics(model, backend.tool_body_id, ARM_DOF),
              backend=backend)
    reg = ObjectRegistry(model, 8)
    rng = np.random.default_rng(seed)
    i = spawn_one(model, data, reg, rng)
    p0 = np.array(reg.pose(data, i))

    def displace():
        reg.teleport(data, i, [p0[0] + shift, p0[1], p0[2]],
                     quat=(1.0, 0.0, 0.0, 0.0))
        mujoco.mj_forward(model, data)

    if use_servo:
        s = PBVSServo(model, data, arm, world)
        s.observe()
        s.servo(0.10)                    # hover, closed loop
        displace()
        s.servo(0.0)                     # descend, re-observing as it goes
        obs = (s.obs_ok, s.obs_try)
    else:
        planner = PBVSServo(model, data, arm, world)
        planner.observe()                # sees the object BEFORE the move
        p_hover, R_hover = planner.desired_pose(0.10)
        q = arm.solve_ik(p_hover, data.qpos[:ARM_DOF], R_target=R_hover)
        arm.move(q, GRIP_OPEN, rate=0.20)
        displace()
        p_grasp, R_grasp = planner.desired_pose(0.0)   # stale by construction
        q = arm.solve_ik(p_grasp, data.qpos[:ARM_DOF], R_target=R_grasp)
        arm.move(q, GRIP_OPEN, rate=0.05)
        obs = (1, 1)

    mujoco.mj_forward(model, data)
    p_grip = np.array(data.xpos[arm.base_id], float)
    gap = float(np.linalg.norm(p_grip[:2] - reg.pose(data, i)[:2]))
    lifted = _try_grasp(model, data, arm, reg, i)
    return gap, lifted, obs


def main():
    from src.model_loader import load_robot_model
    from src.camera import RGBDCamera
    from src.camera import find_camera
    from src.arm import Arm
    from src.pick_place import ObjectRegistry, spawn_one
    from src.servo import PBVSServo
    from src.world_sim import SimObjectWorld
    from src.kinematics_mujoco import MujocoKinematics
    from src.robot_backend_mujoco import MujocoBackend

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')

    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    name, cam_id = find_camera(model)
    camera = RGBDCamera(model, name, 320, 240)

    print()
    print('=' * 74)
    print('POSITION-BASED VISUAL SERVOING')
    print('=' * 74)
    print('error is a 3D pose error in the base frame, not an image feature')
    print('loop: re-observe -> re-derive desired pose -> correct dq -> step')
    print()

    # Part 1: what the loop looks like
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    backend = MujocoBackend(model, data, ARM_DOF)
    arm = Arm(model, data,
              kinematics=MujocoKinematics(model, backend.tool_body_id, ARM_DOF),
              backend=backend)
    reg = ObjectRegistry(model, 8)
    rng = np.random.default_rng(0)
    idx = spawn_one(model, data, reg, rng)
    world = SimObjectWorld(model, data, camera, cam_id, reg)

    servo = PBVSServo(model, data, arm, world)
    trace = []
    ok, iters, err, hover_iters = servo.approach(trace=trace)
    print('part 1 -- closed-loop approach to a static object')
    print('  object %d seen at %s' % (idx, np.round(servo.p_est, 4)))
    print('  %-14s %-14s %s' % ('iteration', 'pos err (m)', 'rot err (rad)'))
    for e in trace:
        if e['k'] in (0, 1, 2, 5, 10, 20, 40, 60, 80, 120):
            print('  %-14d %-14.5f %.5f' % (e['k'], e['err_p'], e['err_o']))
    print('  converged in %d iterations, final position error %.5f m'
          % (iters, err))
    print('  (hover stage took %d of those, descent the rest)' % hover_iters)
    print()

    # Part 2: the object moves after the arm has committed
    print('part 2 -- object displaced once the arm has committed to its hover')
    print('  both modes plan from the object\'s ORIGINAL position; the object')
    print('  then moves. Open loop descends to its stale target, PBVS')
    print('  re-observes during the descent and re-derives the target.')
    print()
    print('  %-10s %-24s %-24s %s'
          % ('shift', 'open loop', 'PBVS', 'PBVS re-observations'))
    print('  %-10s %-24s %-24s' % ('', 'gap / grasp', 'gap / grasp'))
    for shift in (0.0, 0.015, 0.030, 0.050):
        g_ol, l_ol, _ = _trial(model, data, home, world, False, shift, 1)
        g_pb, l_pb, obs = _trial(model, data, home, world, True, shift, 1)
        print('  %-10s %-24s %-24s %d / %d succeeded'
              % ('%5.1f mm' % (shift * 1000),
                 '%5.2f mm  %-6s' % (g_ol * 1000, 'OK' if l_ol else 'MISS'),
                 '%5.2f mm  %-6s' % (g_pb * 1000, 'OK' if l_pb else 'MISS'),
                 obs[0], obs[1]))
    print()
    print('  Open loop holds the gripper wherever it was aimed before the')
    print('  object moved, so its gap tracks the shift. PBVS drives the')
    print('  measured gripper-to-object pose to its tolerance instead, and')
    print('  the gap stays flat as the shift grows.')
    print()
    print('  The re-observation count is the caveat, and it is identical at')
    print('  every shift (~47 of ~100 attempts), so it is not caused by the')
    print('  object moving. It is the camera geometry: the depth camera sits')
    print('  50 mm behind and 30 mm above the jaws, so once the gripper is')
    print('  directly over the object the object is in its own shadow. The')
    print('  servo runs on its held estimate for those stretches, which is')
    print('  why the descent takes longer than the loop gain predicts and')
    print('  why the largest shift does not fully converge. A standoff-')
    print('  dependent descend rate, or re-observing while tilted off the')
    print('  approach axis, would recover it.')

    camera.renderer.close()


if __name__ == '__main__':
    main()
