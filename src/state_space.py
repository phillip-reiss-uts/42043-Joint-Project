# State-space model of the manipulator, with controllability/observability.
#
# KINEMATICS: M(q), g(q) and the servo gains come from a Kinematics provider
# (src/kinematics.py) when one is supplied. The `(model, data)` signatures are
# still accepted so existing callers keep working; passing a provider is what a
# hardware build does, and it removes the direct MuJoCo dependency from the
# linearisation path.
#
# NOTE: mass_matrix/gravity_torque have no clean UR3e equivalent. A real
# implementation needs a pinned or system-identified model; see kinematics.py.

import numpy as np

from src.config import ARM_DOF


def _default_kin(model, data):
    # Build a MuJoCo provider for the legacy call style.
    import mujoco
    from src.kinematics import MujocoKinematics
    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    return MujocoKinematics(model, tool, ARM_DOF)


def _resolve(model, data, kin):
    # Accept either a provider or the legacy (model, data) pair.
    if kin is not None:
        return kin
    return _default_kin(model, data)


# PLANT
#
# The former forward_minimal() -- mj_kinematics + mj_comPos + mj_crb +
# mj_factorM + mj_rne -- has moved into MujocoKinematics._forward_dynamics
# (src/kinematics.py). It had no remaining callers once the mass/gravity
# queries were routed through the provider, and keeping it here would have left
# a direct MuJoCo call inside the portable layer.


def mass_matrix(model=None, data=None, q=None, kin=None):
    # M(q) for the arm joints, including actuator armature.
    return _resolve(model, data, kin).mass_matrix(np.asarray(q, float))


def gravity_torque(model=None, data=None, q=None, kin=None):
    # g(q): joint torque required to hold q against gravity.
    return _resolve(model, data, kin).gravity_torque(np.asarray(q, float))


def gravity_gradient(q, kin=None, h=1e-6, model=None, data=None):
    # G = dg/dq, by central differences on the gravity torque.
    k = _resolve(model, data, kin)
    G = np.zeros((ARM_DOF, ARM_DOF))
    for j in range(ARM_DOF):
        qp = np.array(q, float)
        qm = np.array(q, float)
        qp[j] += h
        qm[j] -= h
        G[:, j] = (k.gravity_torque(qp) - k.gravity_torque(qm)) / (2.0 * h)
    return G


def servo_gains(model=None, kin=None):
    # Per-joint (Kp, Kv). From the provider when available, so a hardware
    # implementation can supply the real controller's stiffness.
    if kin is not None:
        return kin.servo_gains()
    return _default_kin(model, None).servo_gains()


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
    G = gravity_gradient(q0, kin=k)
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


# STRUCTURAL PROPERTIES
def controllability_matrix(A, B):
    # Kalman controllability matrix [B, AB, A^2B, ...]. Reported for reference.
    n = A.shape[0]
    blocks = [B]
    for _ in range(1, n):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def observability_matrix(A, C):
    # Kalman observability matrix [C ; CA ; CA^2 ; ...]. Reported for reference.
    n = A.shape[0]
    blocks = [C]
    for _ in range(1, n):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def pbh_controllability(A, B):
    # PBH test: min over eigenvalues of rank[A - lambda I, B].
    n = A.shape[0]
    eye = np.eye(n)
    ranks = [np.linalg.matrix_rank(np.hstack([A - lam * eye, B]))
             for lam in np.linalg.eigvals(A)]
    return min(ranks), ranks


def pbh_observability(A, C):
    # PBH test: min over eigenvalues of rank[[A - lambda I]; [C]].
    n = A.shape[0]
    eye = np.eye(n)
    ranks = [np.linalg.matrix_rank(np.vstack([A - lam * eye, C]))
             for lam in np.linalg.eigvals(A)]
    return min(ranks), ranks


def position_output(n=ARM_DOF):
    # C for joint encoders: all 6 angles measured, no velocities.
    return np.hstack([np.eye(n), np.zeros((n, n))])


def lqr_gain(A, B, Q, R):
    # Continuous-time LQR gain: minimises int (x'Qx + u'Ru) dt.
    from scipy.linalg import solve_continuous_are
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def analyse(model, data, q0, label='home'):
    # Print the full state-space analysis at one operating point.
    lin = linearise(model, data, q0)
    n = ARM_DOF
    A_tau, B_tau = lin['A_tau'], lin['B_tau']
    A_cl, B_cl = lin['A_cl'], lin['B_cl']
    C_pos = position_output()

    print('=' * 74)
    print('OPERATING POINT: %s' % label)
    print('  q0 = %s' % np.round(lin['q0'], 4))
    print('  g(q0) = %s N.m   (gravity trim)' % np.round(lin['g0'], 2))
    print('=' * 74)
    print('state x = [q ; q_dot]   dim %d        input dim %d' % (2 * n, n))
    print()
    print('M(q) diagonal      : %s' % np.round(np.diag(lin['M']), 4))
    print('M(q) symmetric PD  : %s   cond %.1f'
          % (bool(np.all(np.linalg.eigvalsh(lin['M']) > 0)),
             np.linalg.cond(lin['M'])))
    print('||G|| = ||dg/dq||  : %.2f' % np.linalg.norm(lin['G']))
    print('servo Kp           : %s' % lin['Kp'])
    print('servo Kv           : %s' % lin['Kv'])
    print()
    ev_tau = np.linalg.eigvals(A_tau)
    ev_cl = np.linalg.eigvals(A_cl)
    print('eig(A_tau)  max real part : %+.3f   <- instability rate (1/s)'
          % ev_tau.real.max())
    print('eig(A_cl)   max real part : %+.3f   <- negative: servos stabilise it'
          % ev_cl.real.max())
    print('eig(-M^-1 Kv) max real    : %+.1f   <- negative: velocity feedback'
          ' is dissipative' % np.linalg.eigvals(-lin['Minv'] @ np.diag(lin['Kv'])).real.max())
    print()

    # CONTROLLABILITY
    # PBH is authoritative; the Kalman matrix is ill-conditioned here.
    print('CONTROLLABILITY                                    (rank / %d)' % (2 * n))
    for name, A, B in (('torque u = tau - g(q0)', A_tau, B_tau),
                       ('command u = q_des     ', A_cl, B_cl)):
        r_pbh, _ = pbh_controllability(A, B)
        Wc = controllability_matrix(A, B)
        r_kal = np.linalg.matrix_rank(Wc)
        print('  %s  PBH %2d  %s' % (name, r_pbh,
              'CONTROLLABLE' if r_pbh == 2 * n else 'NOT controllable'))
        print('  %s  Kalman %2d (cond %.1e -- unreliable here)'
              % (' ' * len(name), r_kal, np.linalg.cond(Wc)))
    print()

    # OBSERVABILITY
    print('OBSERVABILITY                                      (rank / %d)' % (2 * n))
    for cname, Cm in (('angles only (q_dot unmeasured)', C_pos),
                      ('angles + velocities           ', np.eye(2 * n))):
        r_pbh, _ = pbh_observability(A_cl, Cm)
        Wo = observability_matrix(A_cl, Cm)
        r_kal = np.linalg.matrix_rank(Wo)
        print('  %s  PBH %2d  %s' % (cname, r_pbh,
              'OBSERVABLE' if r_pbh == 2 * n else 'NOT observable'))
        print('  %s  Kalman %2d (cond %.1e)' % (' ' * len(cname), r_kal,
                                                 np.linalg.cond(Wo)))
    print()
    return lin
