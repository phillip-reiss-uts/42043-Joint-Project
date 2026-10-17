# Regression test: analytic M(q)/g(q) must match the simulator.
#
# This is the check that makes `AnalyticKinematics` trustworthy as a hardware
# replacement for `MujocoKinematics`. It caught three real bugs during
# development (merged gripper mass, merged gripper COM, and a wrong cross-term
# formula), so keep it.
#
# Run: python tests/test_dynamics.py

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOL = 1e-10          # machine-precision agreement expected


def main():
    import mujoco
    from src.model_loader import load_robot_model
    from src.arm_dynamics import ArmDynamics
    from src.kinematics import MujocoKinematics

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    mj = MujocoKinematics(model, tool, 6)
    dyn = ArmDynamics.from_mujoco(model, 6)

    rng = np.random.default_rng(0)
    configs = [np.array(data.qpos[:6])] + [rng.uniform(-2.5, 2.5, 6)
                                           for _ in range(8)]

    worst_M = worst_g = 0.0
    for q in configs:
        Mt, gt = mj.mass_gravity(q)
        Mp, gp = dyn.mass_gravity(q)
        worst_M = max(worst_M, float(np.max(np.abs(Mt - Mp))))
        worst_g = max(worst_g, float(np.max(np.abs(gt - gp))))

    print()
    print('=' * 70)
    print('ANALYTIC DYNAMICS vs MUJOCO')
    print('=' * 70)
    print('  configurations checked : %d' % len(configs))
    print('  rotor inertia          : %s' % np.round(dyn.rotor, 4))
    print('  worst |dM|             : %.3e' % worst_M)
    print('  worst |dg|             : %.3e' % worst_g)
    print()

    # M must also be symmetric and positive definite.
    q = configs[1]
    M = dyn.mass_matrix(q)
    sym = float(np.max(np.abs(M - M.T)))
    eig = np.linalg.eigvalsh(M)
    print('  symmetry |M - M^T|     : %.3e' % sym)
    print('  min eigenvalue of M    : %.6f  (%s)'
          % (eig.min(), 'positive definite' if eig.min() > 0 else 'NOT PD'))
    print()

    ok = worst_M < TOL and worst_g < TOL and sym < 1e-12 and eig.min() > 0
    if ok:
        print('PASS: analytic dynamics reproduces the simulator exactly.')
    else:
        print('FAIL: dynamics diverge from the simulator.')
    print('=' * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
