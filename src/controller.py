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


# DEMO
def main():
    # Show what closing the loop on the estimate buys over open-loop moves.
    import mujoco
    from src.model_loader import load_robot_model
    from src.observer import ArmObserver
    from src.config import GRIP_OPEN

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')

    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    GRIP = model.body('gripper/base').id

    def trial(feedback, Kq=FB_KQ, force=30.0, n_steps=1400, lqr=None):
        # Settle at a pose, then push the tip and measure steady deflection.
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, home)
        mujoco.mj_forward(model, data)
        q0 = np.array(data.qpos[:ARM_DOF])

        obs = ArmObserver(model, data, q0,
                          lqr_weights=(lqr_weights() if lqr is not None
                                       else None))
        ctl = LQRController(obs) if lqr is not None else JointController(Kq=Kq)
        enc = Encoder()

        target = q0.copy()
        target[1] += 0.25
        target[2] -= 0.20
        settle = int(n_steps * 0.6)
        push_at = int(n_steps * 0.75)

        defl = []
        for k in range(n_steps):
            a = min(1.0, k / settle)
            a = a * a * (3.0 - 2.0 * a)
            q_ref = q0 + a * (target - q0)

            if feedback == 'open':
                u = q_ref
            else:
                u = ctl.command(q_ref, np.zeros(ARM_DOF), obs.xhat)

            obs.predict(u)
            data.ctrl[:ARM_DOF] = u
            data.ctrl[model.nu - 1] = GRIP_OPEN
            data.xfrc_applied[GRIP] = ([force, 0.0, 0.0, 0.0, 0.0, 0.0]
                                       if k >= push_at else [0.0] * 6)
            mujoco.mj_step(model, data)
            obs.update(enc.read(data))

            if k >= n_steps - 100:            # settled under the load
                defl.append(np.linalg.norm(data.qpos[:ARM_DOF] - q_ref))
        return float(np.mean(defl)), obs

    print()
    print('=' * 74)
    print('FEEDBACK CONTROL ON ESTIMATED STATE')
    print('=' * 74)
    print('inner loop : MuJoCo position servo (Kp, Kv), unchanged')
    print('feedback   : observer estimate only; data.qvel never read')
    print('disturbance: 30 N world-frame push at gripper/base (xfrc_applied)')
    print()

    base, obs_base = trial('open')
    pd, _ = trial('pd', Kq=FB_KQ)
    lq, obs_lqr = trial('lqr', lqr=lqr_weights())

    print('  %-34s %-16s %s' % ('outer loop', 'deflection (rad)', 'vs open'))
    print('  %-34s %-16.5f %s' % ('open loop (servo alone)', base,
                                    '(reference)'))
    print('  %-34s %-16.5f %.2fx less' % ('PD  (hand-tuned Kq=%.0f)'
                                          % FB_KQ, pd, base / pd))
    print('  %-34s %-16.5f %.2fx less' % ('LQR (designed from model)', lq,
                                           base / lq))
    print()

    for tag, o in (('servo alone', obs_base), ('with LQR outer loop', obs_lqr)):
        ev = np.linalg.eigvals(o.A if tag == 'servo alone'
                               else (o.A - o.B @ o.K_lqr))
        wn = np.abs(ev)
        zeta = -ev.real / np.maximum(wn, 1e-12)
        print('  %-22s maxRe %+8.3f 1/s   min damping ratio %.2f'
              % (tag, ev.real.max(), zeta.min()))
    print()
    print('  Both outer loops reduce the deflection. The difference is where')
    print('  the numbers come from: PD is FB_KQ/FB_TD typed into config, LQR')
    print('  is K = R^-1 B\'P from the Riccati equation, recomputed at every')
    print('  re-linearisation so it tracks the arm\'s configuration.')
    print()
    print('  LQR gain at the final pose: %s'
          % LQRController(obs_lqr).gain_summary())


if __name__ == '__main__':
    main()
