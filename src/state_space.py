"""Linearised state-space model of the manipulator: plant A/B, position output C
and the LQR gain the observer uses.

M(q), g(q) and the servo gains come from a Kinematics provider when one is
supplied; the (model, data) pair is only the legacy fallback, which builds a
MuJoCo provider.
"""

import numpy as np

from src.config import ARM_DOF


def _default_kin(model, data):
    # MuJoCo provider for the legacy (model, data) call style.
    import mujoco
    from src.kinematics_mujoco import MujocoKinematics
    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    return MujocoKinematics(model, tool, ARM_DOF)


def _resolve(model, data, kin):
    # Accept either a provider or the legacy (model, data) pair.
    if kin is not None:
        return kin
    return _default_kin(model, data)


def linearise(model, data, q0, kin=None):
    # Linearise about the equilibrium (q = q0, q' = 0).
    #
    # The algebra below is pure numpy; only the plant quantities M, G, g and
    # the servo gains come from the provider, so a hardware build can linearise
    # around a real controller's model without changing this function.
    k = _resolve(model, data, kin)
    q0 = np.asarray(q0, float)[:ARM_DOF]
    n = ARM_DOF
    Z, I = np.zeros((n, n)), np.eye(n)

    M = k.mass_matrix(q0)
    # G = dg/dq, central difference on the provider's g(q).
    h = 1e-6
    G = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = h
        G[:, j] = (k.gravity_torque(q0 + dq) - k.gravity_torque(q0 - dq)) / (2.0 * h)
    Minv = np.linalg.inv(M)
    Kp, Kv = k.servo_gains()
    KP, KV = np.diag(Kp), np.diag(Kv)
    g0 = k.gravity_torque(q0)

    # torque input, referred to the gravity trim g(q0)
    A_tau = np.block([[Z, I], [-Minv @ G, Z]])
    B_tau = np.vstack([Z, Minv])

    # position-command input. NOTE: Kp/Kv describe the *simulated* position
    # servo. On hardware these come from the real controller (see
    # MujocoKinematics.servo_gains), so A_cl changes meaning there.
    A_cl = np.block([[Z, I], [-Minv @ (KP + G), -Minv @ KV]])
    B_cl = np.vstack([Z, Minv @ KP])

    return dict(q0=q0, M=M, G=G, Kp=Kp, Kv=Kv, Minv=Minv, g0=g0,
                A_tau=A_tau, B_tau=B_tau, A_cl=A_cl, B_cl=B_cl)


def position_output(n=ARM_DOF):
    # C for joint encoders: all 6 angles measured, no velocities.
    return np.hstack([np.eye(n), np.zeros((n, n))])


def lqr_gain(A, B, Q, R):
    # Continuous-time LQR gain: minimises int (x'Qx + u'Ru) dt.
    from scipy.linalg import solve_continuous_are
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)

