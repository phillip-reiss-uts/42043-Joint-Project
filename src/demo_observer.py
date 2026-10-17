# Arm state-observer demonstration.
#
# DEMONSTRATION ONLY. Needs the composed MuJoCo scene, so it lives outside
# observer.py. See tests/test_layering.py.
#
#     python -m src.demo_observer

import numpy as np
import mujoco

from src.config import ARM_DOF, ENCODER_NOISE, TAU_NOISE, OBSERVER_RELIN, GRIP_OPEN
from src.observer import ArmObserver


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

    obs = ArmObserver(model, data, q_home)
    print()
    print('=' * 74)
    print('ARM STATE OBSERVER  (extended Kalman filter, joint angles only)')
    print('=' * 74)
    print('state dim        : %d  (6 angles + 6 velocities)'
          % (2 * ARM_DOF))
    print('measured         : %d  (angles only; velocity unmeasured)'
          % ARM_DOF)
    print('timestep         : %.4f s' % model.opt.timestep)
    print('encoder sigma    : %.1e rad     torque sigma: %.2f N.m'
          % (ENCODER_NOISE, TAU_NOISE))
    print('prediction       : nonlinear M(q)q\'\' = Kp(u-q) - Kv q\' - g(q)')
    print('re-linearise     : every %d steps (%.2f s)'
          % (OBSERVER_RELIN, OBSERVER_RELIN * model.opt.timestep))
    print()

    dt = model.opt.timestep
    rng = np.random.default_rng(0)

    # Phase 1: hold still, but start the observer with a wrong velocity.
    print('PHASE 1 -- convergence from a wrong velocity guess')
    print('  arm held at home (true velocity = 0), observer initialised')
    print('  believing all joints move at +0.50 rad/s')
    print()
    obs.xhat[ARM_DOF:] = 0.5
    hold_err = []
    for k in range(400):
        u = q_home.copy()
        obs.predict(u)
        data.ctrl[:ARM_DOF] = u
        mujoco.mj_step(model, data)
        y = data.qpos[:ARM_DOF] + rng.normal(0.0, ENCODER_NOISE, ARM_DOF)
        obs.update(y)
        hold_err.append(np.linalg.norm(obs.qdot - data.qvel[:ARM_DOF]))
    print('  %-8s %-14s %s' % ('step', 'time (s)', '|v_hat - v_true| (rad/s)'))
    for k in (0, 5, 10, 20, 40, 80, 200, 399):
        print('  %-8d %-14.3f %.5f' % (k, k * dt, hold_err[k]))
    print()

    # Phase 2: drive a slow sinusoid and compare against differencing.
    print('PHASE 2 -- tracking a moving arm, vs differencing the encoders')
    n_steps = 1500                       # 3 s
    t = np.arange(n_steps) * dt
    joints = ((1, 0.15), (2, 0.12), (3, 0.10), (4, 0.08))

    truth_q, truth_v, est_q, est_v, naive_v = [], [], [], [], []
    prev_y = None

    for k in range(n_steps):
        u = q_home.copy()
        for j, amp in joints:
            u[j] = q_home[j] + amp * np.sin(2 * np.pi * 0.25 * t[k])

        obs.predict(u)                   # propagate to k+1
        data.ctrl[:ARM_DOF] = u
        mujoco.mj_step(model, data)      # the real (nonlinear) plant

        y = data.qpos[:ARM_DOF] + rng.normal(0.0, ENCODER_NOISE, ARM_DOF)
        obs.update(y)

        truth_q.append(data.qpos[:ARM_DOF].copy())
        truth_v.append(data.qvel[:ARM_DOF].copy())
        est_q.append(obs.q)
        est_v.append(obs.qdot)
        naive_v.append((y - prev_y) / dt if prev_y is not None
                       else np.zeros(ARM_DOF))
        prev_y = y.copy()

    truth_q = np.array(truth_q); truth_v = np.array(truth_v)
    est_q = np.array(est_q);     est_v = np.array(est_v)
    naive_v = np.array(naive_v)

    s = 100                              # skip the transient
    pos_err = np.linalg.norm(est_q[s:] - truth_q[s:], axis=1)
    vel_err = np.linalg.norm(est_v[s:] - truth_v[s:], axis=1)
    np_err = np.linalg.norm(naive_v[s:] - truth_v[s:], axis=1)
    true_v_mag = np.linalg.norm(truth_v[s:], axis=1).mean()

    print()
    print('  true velocity magnitude (mean)  : %.4f rad/s' % true_v_mag)
    print()
    print('  %-38s %-11s %s' % ('', 'mean', 'max'))
    print('  %-38s %-11.5f %.5f' % ('angle error (rad)', pos_err.mean(),
                                    pos_err.max()))
    print('  %-38s %-11.5f %.5f' % ('velocity error, observer (rad/s)',
                                    vel_err.mean(), vel_err.max()))
    print('  %-38s %-11.5f %.5f' % ('velocity error, finite diff (rad/s)',
                                    np_err.mean(), np_err.max()))
    print()
    print('  velocity accuracy: %.1f%% of the signal'
          % (100 * (1 - vel_err.mean() / true_v_mag)))
    print('  observer is %.1fx better than differencing the encoders'
          % (np_err.mean() / max(vel_err.mean(), 1e-12)))
    print()

    ev = obs.error_dynamics()
    print('=' * 74)
    print('observer error dynamics  eig((I - K C) A_d): max |eig| = %.4f  %s'
          % (np.abs(ev).max(),
             'STABLE' if np.abs(ev).max() < 1 else 'UNSTABLE'))
    print('steady-state P diagonal, velocity block (variance):')
    print(np.round(np.diag(obs.P)[ARM_DOF:], 8))
    print()

    # Phase 3: how each method degrades as the encoder gets noisier.
    print('=' * 74)
    print('PHASE 3 -- velocity error vs encoder noise')
    print('=' * 74)
    print('  %-12s %-16s %-16s %s'
          % ('encoder std', 'observer err', 'finite-diff err', 'improvement'))
    for sq in (0.0, 1.0e-5, 1.0e-4, 1.0e-3):
        o, n = _run_trial(model, data, home, q_home, sq, n_steps=1200)
        print('  %-12.1e %-16.5f %-16.5f %.1fx'
              % (sq, o, n, (n / o) if o > 0 else float('inf')))
    print()
    print('  At zero noise the observer sits at its integration floor (~0.01')
    print('  rad/s) and differencing is exact, as expected. As encoder noise')
    print('  rises the observer barely moves while differencing grows without')
    print('  bound -- differencing amplifies noise by 1/dt = 500.')
    print('=' * 74)


def _run_trial(model, data, home, q_home, sigma_q, n_steps=1200):
    # One closed-loop trial at a given encoder noise. Returns (obs_err, naive_err).
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    obs = ArmObserver(model, data, q_home, sigma_q=sigma_q)
    rng = np.random.default_rng(0)
    dt = model.opt.timestep
    t = 0.0
    joints = ((1, 0.15), (2, 0.12), (3, 0.10), (4, 0.08))
    ve, nv = [], []
    prev_y = None
    for k in range(n_steps):
        u = q_home.copy()
        for j, amp in joints:
            u[j] = q_home[j] + amp * np.sin(2 * np.pi * 0.25 * t)
        obs.predict(u)
        data.ctrl[:ARM_DOF] = u
        mujoco.mj_step(model, data)
        y = data.qpos[:ARM_DOF] + rng.normal(0.0, sigma_q, ARM_DOF)
        obs.update(y)
        if k > 100:
            v_true = data.qvel[:ARM_DOF]
            ve.append(np.linalg.norm(obs.qdot - v_true))
            if prev_y is not None:
                nv.append(np.linalg.norm((y - prev_y) / dt - v_true))
        prev_y = y.copy()
        t += dt
    return float(np.mean(ve)), float(np.mean(nv)) if nv else float('nan')


if __name__ == '__main__':
    main()
