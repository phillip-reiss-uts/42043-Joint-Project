# State observer for the arm: estimates joint velocity from position encoders.
#
# KINEMATICS: M(q), g(q) and the servo gains come from a Kinematics provider
# (src/kinematics.py), injected via `kin=`. The default builds a MuJoCo one, so
# no behaviour changes, but this module no longer imports mujoco at top level.

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from src.config import (
    ARM_DOF, ENCODER_NOISE, TAU_NOISE, OBSERVER_RELIN, OBSERVER_SUBSTEPS,
)
from src.state_space import linearise, position_output, lqr_gain


# DISCRETE
def discretise(A, B, dt):
    # Exact zero-order-hold discretisation: (A_d, B_d) for sampling dt.
    n, m = B.shape
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A
    aug[:n, n:] = B
    E = expm(aug * dt)
    return E[:n, :n], E[:n, n:]


def process_noise(A, Gq, dt, n_sub=16):
    # Q_d = int_0^dt exp(A s) Gq Gq' exp(A' s) ds, by midpoint quadrature.
    Qc = Gq @ Gq.T
    n = A.shape[0]
    ds = dt / n_sub
    Qd = np.zeros((n, n))
    for i in range(n_sub):
        s = (i + 0.5) * ds
        E = expm(A * s)
        Qd += (E @ Qc @ E.T) * ds
    return Qd


def kalman_gain(A_d, C, Q_d, R):
    # Steady-state discrete Kalman gain from the DARE. Returns (K, P).
    P = solve_discrete_are(A_d.T, C.T, Q_d, R)
    S = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S)
    return K, P


# OBSERVER
class ArmObserver:
    # Estimates [q ; q'] for the 6 arm joints from joint-angle measurements.

    def __init__(self, model, data, q0, sigma_q=ENCODER_NOISE,
                 sigma_tau=TAU_NOISE, relin_steps=OBSERVER_RELIN,
                 substeps=OBSERVER_SUBSTEPS, lqr_weights=None, kin=None):
        self.model = model
        self.data = data
        self.n = ARM_DOF
        self.C = position_output()
        self.R = (sigma_q ** 2) * np.eye(self.n)
        self.sigma_tau = sigma_tau
        self.relin_steps = relin_steps
        self.substeps = substeps
        self.dt = model.opt.timestep
        self.lqr_weights = lqr_weights

        # Plant model from the injected provider (src/kinematics.py), so the
        # observer does not reach into the simulator for M(q)/g(q).
        if kin is None:
            import mujoco
            from src.kinematics_mujoco import MujocoKinematics
            tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     'gripper/base')
            kin = MujocoKinematics(model, tool, self.n)
        self.kin = kin

        Kp, Kv = kin.servo_gains()
        self.Kp, self.Kv = Kp, Kv

        self.xhat = np.zeros(2 * self.n)
        self.xhat[:self.n] = np.asarray(q0, float)[:self.n]

        # velocity is unmeasured, so start with a wide prior on it
        self.P = np.diag(np.r_[np.full(self.n, sigma_q ** 2),
                               np.full(self.n, 1.0)])
        self.n_since_relin = 0
        self.relin(self.xhat[:self.n], reset_cov=True)

    # MODEL
    def _mass_gravity(self, q):
        # M(q) and g(q) for the arm joints, from the provider. One pass for
        # both, which is why Kinematics.mass_gravity exists.
        M, g = self.kin.mass_gravity(np.asarray(q, float))
        return M[:self.n, :self.n], g[:self.n]

    def relin(self, q, reset_cov=False):
        # Refresh A_d, B_d, Q_d (and the steady-state gain) for configuration q.
        lin = linearise(self.model, self.data, q, kin=self.kin)
        self.A, self.B = lin['A_cl'], lin['B_cl']
        self.Minv = lin['Minv']
        self.A_d, self.B_d = discretise(self.A, self.B, self.dt)

        Gq = np.vstack([np.zeros((self.n, self.n)), self.Minv]) * self.sigma_tau
        self.Q_d = process_noise(self.A, Gq, self.dt)

        self.K_ss, P_ss = kalman_gain(self.A_d, self.C, self.Q_d, self.R)
        if self.lqr_weights is not None:
            self.K_lqr = lqr_gain(self.A, self.B, *self.lqr_weights)
        if reset_cov:
            self.P = P_ss
        self.n_since_relin = 0

    # FILTER
    def predict(self, u):
        # Propagate one step under joint-position command u (rad).
        u = np.asarray(u, float)
        q = self.xhat[:self.n].copy()
        v = self.xhat[self.n:].copy()
        M, g = self._mass_gravity(q)

        h = self.dt / self.substeps
        for _ in range(self.substeps):
            acc = np.linalg.solve(M, self.Kp * (u - q) - self.Kv * v - g)
            v = v + acc * h
            q = q + v * h
        self.xhat = np.r_[q, v]

        self.P = self.A_d @ self.P @ self.A_d.T + self.Q_d

        self.n_since_relin += 1
        if self.n_since_relin >= self.relin_steps:
            self.relin(self.xhat[:self.n], reset_cov=False)

    def update(self, y):
        # Correct with a joint-angle measurement y (rad).
        y = np.asarray(y, float)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.xhat = self.xhat + K @ (y - self.C @ self.xhat)
        self.P = (np.eye(2 * self.n) - K @ self.C) @ self.P

    # ACCESSORS
    @property
    def q(self):
        return self.xhat[:self.n].copy()

    @property
    def qdot(self):
        return self.xhat[self.n:].copy()

    def error_dynamics(self):
        # Eigenvalues of the observer error map (I - K C) A_d.
        return np.linalg.eigvals((np.eye(2 * self.n) - self.K_ss @ self.C)
                                 @ self.A_d)
