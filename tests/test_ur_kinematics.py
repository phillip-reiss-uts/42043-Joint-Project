"""Regression test: the HARDWARE kinematics must match the simulator.

`src/kinematics_ur.py` is what a real UR3e build uses: FK/Jacobian from the
link chain in src/arm_dynamics.py, M(q)/g(q) from the same. The chain numbers
are derived from the arm + payload link data, which is a property of the
hardware, not of MuJoCo -- but MuJoCo is the only independent implementation
of the same geometry available here, so it is the reference.

This runs OFFLINE. No robot, no ur_rtde: it proves that the code path a
hardware front-end would use computes the same FK, Jacobian, M and g as the
simulation, so a porting bug shows up here rather than on the arm.

Run: python tests/test_ur_kinematics.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOL = 1e-10

# The mass matrix gets its own, looser tolerance, and only the mass matrix.
#
# It is the one quantity where the two implementations take different ROUTES to
# the same number: MuJoCo uses a composite rigid-body recursion, the analytic
# chain sums link contributions directly. Different summation orders differ by
# roughly n * eps * cond, so the agreement here is rounding-limited, not exact.
#
# Measured, and both numbers matter:
#     mujoco 3.15   worst |dM| 6.7e-16    worst |dg| 1.1e-14
#     mujoco 3.10   worst |dM| 1.27e-09   (version drift in how MuJoCo
#                                          assembles M, not a standing error)
# TOL_M = 1e-8 covers the older build without hiding a real fault. What this
# test actually caught was a STALE PAYLOAD -- arm_dynamics.json left behind by a
# gripper-mesh change -- and that landed at |dM| 6.6e-3 and |dg| 8.0e-2, ten
# orders above this tolerance. The fix is always to rebuild the data, never to
# loosen the number:  python tools/build_arm_dynamics.py
TOL_M = 1e-8


def main():
    import mujoco
    from src.model_loader import load_robot_model
    from src.arm_dynamics import ArmDynamics
    from src.kinematics_mujoco import MujocoKinematics
    from src.kinematics_ur import URKinematics

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    ref = MujocoKinematics(model, tool, 6)
    # The chain the HARDWARE build uses: shipped link data, no MuJoCo.
    dyn = ArmDynamics.from_file(os.path.join(ROOT, 'models', 'ur3e',
                                             'arm_dynamics.json'))

    # The gripper mount: wrist_3_link -> gripper/base. On the robot this is a
    # measured constant; here it is derived from the composed model so the test
    # exercises the same code path with the true transform.
    w3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist_3_link')
    Rw = np.array(data.xmat[w3], float).reshape(3, 3)
    pw = np.array(data.xpos[w3], float)
    Rg = np.array(data.xmat[tool], float).reshape(3, 3)
    pg = np.array(data.xpos[tool], float)
    tool_pos = Rw.T @ (pg - pw)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (Rw.T @ Rg).reshape(-1))

    # Gains do not affect FK/Jacobian/M/g; any positive pair will do here.
    #
    # root_yaw: model_loader places the arm in the CELL frame, rotating the base
    # by UR_ROOT_YAW so that the table and the measured home pose line up. The
    # chain has to apply the same rotation or it describes the unrotated model
    # and every pose is mirrored -- which is exactly the bug this test exists to
    # catch, so it is not optional here.
    from src.config import UR_ROOT_YAW
    ur = URKinematics(dyn, (np.full(6, 1000.0), np.full(6, 200.0)),
                      tool_pos=tool_pos, tool_quat=quat, root_yaw=UR_ROOT_YAW)

    rng = np.random.default_rng(0)
    configs = [np.array(data.qpos[:6])] + [rng.uniform(-2.5, 2.5, 6)
                                           for _ in range(9)]

    worst_p = worst_R = worst_J = worst_M = worst_g = 0.0
    for q in configs:
        p_ref, R_ref = ref.pose(q)
        p_ur, R_ur = ur.pose(q)
        worst_p = max(worst_p, float(np.max(np.abs(p_ur - p_ref))))
        worst_R = max(worst_R, float(np.max(np.abs(R_ur - R_ref))))

        worst_J = max(worst_J, float(np.max(np.abs(ur.jacobian(q)
                                                   - ref.jacobian(q)))))

        M_ref, g_ref = ref.mass_gravity(q)
        M_ur, g_ur = ur.mass_gravity(q)
        worst_M = max(worst_M, float(np.max(np.abs(M_ur - M_ref))))
        worst_g = max(worst_g, float(np.max(np.abs(g_ur - g_ref))))

    print()
    print('=' * 70)
    print('HARDWARE KINEMATICS (chain) vs MUJOCO')
    print('=' * 70)
    print('  configurations checked : %d' % len(configs))
    print('  worst |dp|             : %.3e m' % worst_p)
    print('  worst |dR|             : %.3e' % worst_R)
    print('  worst |dJ|             : %.3e' % worst_J)
    print('  worst |dM|             : %.3e' % worst_M)
    print('  worst |dg|             : %.3e' % worst_g)
    print()

    checks = [('|dp|', worst_p, TOL),
              ('|dR|', worst_R, TOL),
              ('|dJ|', worst_J, TOL),
              ('|dM|', worst_M, TOL_M),
              ('|dg|', worst_g, TOL)]
    bad = [(name, value, tol) for name, value, tol in checks if value > tol]
    if bad:
        print('FAIL: hardware kinematics differs from the reference:')
        for name, value, tol in bad:
            print('  %-6s %.3e   tolerance %.0e' % (name, value, tol))
        return 1
    print('PASS: the hardware kinematics chain reproduces the simulator.')
    print('      worst |dM| %.2e is within its own tolerance %.0e, and every'
          % (worst_M, TOL_M))
    print('      other quantity agrees to %.0e or better.' % TOL)
    print('=' * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
