# Feedback control running on the observer's estimated state.

import numpy as np

from src.config import (
    ARM_DOF, FB_KQ, FB_TD, ENCODER_NOISE,
    LQR_Q_POS, LQR_Q_VEL, LQR_R,
)


class JointController:
    # PD feedback on the estimated joint state, added to the reference.

    def __init__(self, Kq=FB_KQ, Td=FB_TD):
        self.Kq = float(Kq)
        self.Td = float(Td)

    def command(self, q_ref, qdot_ref, xhat):
        # Return the joint-position command for this step.
        q_hat = xhat[:ARM_DOF]
        qd_hat = xhat[ARM_DOF:]
        return (np.asarray(q_ref, float)
                + self.Kq * (np.asarray(q_ref, float) - q_hat)
                + self.Td * (np.asarray(qdot_ref, float) - qd_hat))

    def describe(self):
        return ('u = q_ref + %.2f (q_ref - q_hat) + %.3f (qdot_ref - qdot_hat)'
                % (self.Kq, self.Td))


class LQRController:
    # Outer loop whose gains come out of the state-space model, not tuning.

    def __init__(self, observer, Q=None, R=None):
        self.observer = observer
        self.Q = Q if Q is not None else lqr_weights()[0]
        self.R = R if R is not None else lqr_weights()[1]

    def command(self, q_ref, qdot_ref, xhat):
        # u = q_ref + K (x_ref - x_hat), with K from the last re-linearisation.
        q_ref = np.asarray(q_ref, float)
        e = np.concatenate([q_ref - xhat[:ARM_DOF],
                            np.asarray(qdot_ref, float) - xhat[ARM_DOF:]])
        return q_ref + self.observer.K_lqr @ e

    def describe(self):
        return ('u = q_ref + K_lqr (x_ref - x_hat),  K from LQR '
                '(Q_pos=%.3g, Q_vel=%.3g, R=%.3g)' % (LQR_Q_POS, LQR_Q_VEL,
                                                      LQR_R))

    def gain_summary(self):
        K = self.observer.K_lqr
        return ('|K_pos| %.2f  |K_vel| %.4f'
                % (np.linalg.norm(K[:, :ARM_DOF]),
                   np.linalg.norm(K[:, ARM_DOF:])))


def lqr_weights():
    # Block-diagonal Q and R for the LQR design.
    Q = np.diag(np.r_[np.full(ARM_DOF, LQR_Q_POS),
                      np.full(ARM_DOF, LQR_Q_VEL)])
    R = LQR_R * np.eye(ARM_DOF)
    return Q, R


class Encoder:
    # Joint-angle measurement: the true angle plus bounded noise.

    def __init__(self, sigma=ENCODER_NOISE, noisy=True, seed=0):
        self.sigma = float(sigma)
        self.noisy = bool(noisy)
        self.rng = np.random.default_rng(seed)

    def read(self, data):
        q = np.array(data.qpos[:ARM_DOF], float)
        if self.noisy and self.sigma > 0.0:
            q = q + self.rng.normal(0.0, self.sigma, ARM_DOF)
        return q
