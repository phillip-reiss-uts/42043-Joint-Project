# Regression test: the two kinematics must agree, in the CELL frame.
#
# WHY THIS EXISTS SEPARATELY FROM test_ur_kinematics.py
# ----------------------------------------------------
# That test compares the chain against the simulator through five quantities at
# once -- position, rotation, Jacobian, mass matrix, gravity. It works, and the
# mass matrix is what actually caught a stale payload, but it is an indirect
# statement about a frame convention: to read it you have to trust that M and g
# were built from the same frames you care about.
#
# This one asserts the convention DIRECTLY, on the number a porting bug moves
# first: the TCP pose. A 180 deg error in the base heading shows up here as a
# plain sign flip on x and y, with no interpretation needed. That is the exact
# bug todo F1 records -- the chain described the unrotated model while the cell
# is turned half a turn -- and it survived several weeks because nothing checked
# the pose itself.
#
# The second check is the important one: with root_yaw left at zero the chain
# MUST disagree, by precisely Rz(UR_ROOT_YAW). A test that only ever passes with
# the right flag cannot tell you the flag is being read.
#
# Runs OFFLINE. No robot and no ur_rtde: both sides are pure maths.
#
# Run: python tests/test_kinematics_parity.py

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Machine-precision agreement: both sides are the same link data, summed
# differently. test_ur_kinematics measures 1.7e-16 on position, so anything at
# 1e-12 is a real disagreement, not rounding.
TOL = 1e-12

NCONFIG = 12


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))
    return bool(ok)


def _tool_from_model(mujoco, model, data, tool_id):
    # wrist_3_link -> gripper/base, read out of the composed model. On the robot
    # this is TOOL_MOUNT_POS/QUAT, a measured constant; deriving it here means
    # this test is about the frames, not about a bracket nobody has measured.
    w3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist_3_link')
    Rw = np.array(data.xmat[w3], float).reshape(3, 3)
    pw = np.array(data.xpos[w3], float)
    Rg = np.array(data.xmat[tool_id], float).reshape(3, 3)
    pg = np.array(data.xpos[tool_id], float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (Rw.T @ Rg).reshape(-1))
    return Rw.T @ (pg - pw), quat


def main():
    import mujoco
    from src.config import UR_ROOT_YAW
    from src.arm_dynamics import ArmDynamics
    from src.kinematics_mujoco import MujocoKinematics
    from src.kinematics_ur import URKinematics
    from src.model_loader import load_robot_model

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    ref = MujocoKinematics(model, tool, 6)

    dyn = ArmDynamics.from_file(os.path.join(ROOT, 'models', 'ur3e',
                                             'arm_dynamics.json'))
    tool_pos, tool_quat = _tool_from_model(mujoco, model, data, tool)
    gains = (np.full(6, 1000.0), np.full(6, 200.0))     # do not affect FK

    def chain(root_yaw):
        return URKinematics(dyn, gains, tool_pos=tool_pos, tool_quat=tool_quat,
                            root_yaw=root_yaw)

    ur = chain(UR_ROOT_YAW)
    ur_unrotated = chain(0.0)

    rng = np.random.default_rng(0)
    configs = [np.array(data.qpos[:6])] + [rng.uniform(-2.5, 2.5, 6)
                                           for _ in range(NCONFIG - 1)]

    worst_p = worst_R = 0.0
    for q in configs:
        p_ref, R_ref = ref.pose(q)
        p_ur, R_ur = ur.pose(q)
        worst_p = max(worst_p, float(np.max(np.abs(p_ur - p_ref))))
        worst_R = max(worst_R, float(np.max(np.abs(R_ur - R_ref))))

    # The same comparison with the base rotation NOT applied. There must be a
    # disagreement, and it must be exactly the rotation config says the cell has.
    cz, sz = np.cos(UR_ROOT_YAW), np.sin(UR_ROOT_YAW)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    worst_flip = 0.0
    for q in configs:
        p_ref, _ = ref.pose(q)
        p_wrong, _ = ur_unrotated.pose(q)
        worst_flip = max(worst_flip, float(np.max(np.abs(Rz @ p_wrong - p_ref))))
    unrotated_error = max(
        float(np.max(np.abs(ur_unrotated.pose(q)[0] - ref.pose(q)[0])))
        for q in configs)

    # The approach axis: at any configuration the tool's z must point where the
    # reference says it does, and at home that is straight down. This is the
    # half of the convention R_DOWN is written against.
    #
    # 1e-4, not 1e-9: TOOL_MOUNT_QUAT carries deliberate 1e-4-scale off-axis
    # terms (0.7071068, -0.0001, 0.0002, 0.7071068), so the tool's z misses
    # world -z by ~5e-6 rad at home. A wrong frame gives an error of order 1,
    # so the check still bites.
    p0, R0 = ref.pose(configs[0])
    down = float(np.max(np.abs(R0[:, 2] - np.array([0.0, 0.0, -1.0]))))

    print()
    print('=' * 70)
    print('KINEMATICS FRAME PARITY  (chain vs MuJoCo, cell frame)')
    print('=' * 70)
    print('  configurations          : %d' % len(configs))
    print('  root_yaw                : %.6f rad' % UR_ROOT_YAW)
    print('  tool frame              : %s' % np.round(tool_pos, 6))
    print('  worst |dp|              : %.3e m' % worst_p)
    print('  worst |dR|              : %.3e' % worst_R)
    print('  unrotated chain |dp|    : %.3e m  (must be large)' % unrotated_error)
    print('  Rz(pi) puts it back     : %.3e' % worst_flip)
    print('  tool +z vs world up     : %.3e  (0 means it points down)' % down)
    print()

    failures = 0
    failures += not check('the chain reproduces the TCP position', worst_p < TOL,
                          'worst %.2e m' % worst_p)
    failures += not check('and the orientation', worst_R < TOL,
                          'worst %.2e' % worst_R)
    failures += not check(
        'dropping root_yaw visibly breaks it',
        unrotated_error > 1e-2,
        '%.1f mm apart -- the flag is load-bearing'
        % (unrotated_error * 1e3))
    failures += not check(
        'and Rz(UR_ROOT_YAW) explains the whole difference',
        worst_flip < TOL,
        'residual %.2e m' % worst_flip)
    failures += not check('the tool axis points down at home', down < 1e-4,
                          'worst component %.2e' % down)

    print()
    print('=' * 70)
    if failures:
        print('  %d check(s) FAILED' % failures)
        print('=' * 70)
        return 1
    print('  The chain and the model agree on the pose itself, in the cell')
    print('  frame, and the agreement is shown to depend on UR_ROOT_YAW.')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
