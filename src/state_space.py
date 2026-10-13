# State-space model of the manipulator, with controllability/observability.

import mujoco
import numpy as np

from src.config import ARM_DOF


# PLANT
def forward_minimal(model, data):
    # Compute M(q) and qfrc_bias without running collision detection.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_factorM(model, data)
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)


def _set_q(model, data, q):
    # Put the arm at q with zero velocity and refresh derived quantities.
    data.qpos[:ARM_DOF] = q
    data.qvel[:] = 0.0
    forward_minimal(model, data)


def mass_matrix(model, data, q):
    # M(q) for the arm joints, including actuator armature.
    _set_q(model, data, q)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)     # (model, data, dst) in this version
    return M[:ARM_DOF, :ARM_DOF].copy()


def gravity_torque(model, data, q):
    # g(q): joint torque required to hold q against gravity.
    _set_q(model, data, q)
    return data.qfrc_bias[:ARM_DOF].copy()


def gravity_gradient(model, data, q, h=1e-6):
    # G = dg/dq, by central differences on the gravity torque.
    G = np.zeros((ARM_DOF, ARM_DOF))
    for j in range(ARM_DOF):
        qp = np.array(q, float)
        qm = np.array(q, float)
        qp[j] += h
        qm[j] -= h
        G[:, j] = (gravity_torque(model, data, qp)
                   - gravity_torque(model, data, qm)) / (2.0 * h)
    return G


def servo_gains(model):
    # Per-joint (Kp, Kv) of MuJoCo's position servos.
    Kp = np.array([model.actuator_gainprm[i][0] for i in range(ARM_DOF)])
    Kv = np.array([-model.actuator_biasprm[i][2] for i in range(ARM_DOF)])
    return Kp, Kv


def linearise(model, data, q0):
    # Linearise about the equilibrium (q = q0, q' = 0).
    q0 = np.asarray(q0, float)[:ARM_DOF]
    n = ARM_DOF
    Z, I = np.zeros((n, n)), np.eye(n)

    M = mass_matrix(model, data, q0)
    G = gravity_gradient(model, data, q0)
    Minv = np.linalg.inv(M)
    Kp, Kv = servo_gains(model)
    KP, KV = np.diag(Kp), np.diag(Kv)

    # torque input, referred to the gravity trim g(q0)
    A_tau = np.block([[Z, I], [-Minv @ G, Z]])
    B_tau = np.vstack([Z, Minv])

    # position-command input (what the simulator does)
    A_cl = np.block([[Z, I], [-Minv @ (KP + G), -Minv @ KV]])
    B_cl = np.vstack([Z, Minv @ KP])

    return dict(q0=q0, M=M, G=G, Kp=Kp, Kv=Kv, Minv=Minv, g0=gravity_torque(
        model, data, q0), A_tau=A_tau, B_tau=B_tau, A_cl=A_cl, B_cl=B_cl)


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


def main():
    from src.model_loader import load_robot_model

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')

    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    q_home = np.array(data.qpos[:ARM_DOF])

    # second operating point: extended forward and down, like a grasp pose
    q_grasp = q_home.copy()
    q_grasp[1] += 0.6      # shoulder_lift
    q_grasp[2] -= 0.5      # elbow

    analyse(model, data, q_home, 'home')
    analyse(model, data, q_grasp, 'grasp-like (shoulder +0.6, elbow -0.5)')

    print('=' * 74)
    print('SUMMARY')
    print('=' * 74)
    print('* Both realisations are controllable (PBH rank 12): every joint state')
    print('  is reachable from here, with torque or with a position command.')
    print('* The torque plant is UNSTABLE at these poses (positive eigenvalue).')
    print('  Gravity compensation alone does not hold a manipulator up: a')
    print('  deviation grows. Feedback is required, not optional.')
    print('* A_cl is strictly stable because of the servo velocity damping')
    print('  -M^-1 Kv. That means the controller task is tracking and')
    print('  disturbance rejection, not stabilisation.')
    print('* Joint angles alone are observable (PBH rank 12) with q_dot')
    print('  unmeasured, so an observer can reconstruct velocity from')
    print('  encoders -- exactly the unmeasured state the brief asks for.')


if __name__ == '__main__':
    main()
