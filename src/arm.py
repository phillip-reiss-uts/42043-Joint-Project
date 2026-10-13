# Arm motion: IK, trajectory execution, gripper, and view repositioning.

import mujoco
import numpy as np

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, R_DOWN,
    RAPID, FEED, RAPID_MIN_STEPS, FEED_MIN_STEPS,
    SYNC_STEPS,
)
from src.overlay import push


class Arm:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
        self.grip_act = model.nu - 1          # fingers_actuator

        # Optional closed loop on an estimated state (see controller.py).
        self.observer = None
        self.controller = None
        self.encoder = None

        # set by main so observe() can push estimator error to the overlay
        self.viewer = None

    # STATE FEEDBACK
    def attach_observer(self, observer, controller=None, encoder=None):
        # Close the loop on an estimated state.
        from src.controller import JointController, Encoder
        self.observer = observer
        self.controller = controller or JointController()
        self.encoder = encoder or Encoder()

    def detach_observer(self):
        # Return to open-loop interpolation.
        self.observer = None
        self.controller = None
        self.encoder = None

    def observe(self):
        # Advance the observer one step using the command currently in ctrl.
        if self.observer is None:
            return
        u = np.array(self.data.ctrl[:ARM_DOF], float)
        self.observer.predict(u)
        self.observer.update(self.encoder.read(self.data))
        push(self.viewer, 'observer',
             'obs err %.5f' % float(np.linalg.norm(
                 self.observer.xhat[:ARM_DOF] - self.data.qpos[:ARM_DOF])),
             'v err %.5f' % float(np.linalg.norm(
                 self.observer.xhat[ARM_DOF:] - self.data.qvel[:ARM_DOF])))

    # IK
    def solve_ik(self, target, q_init, R_target=R_DOWN, iters=600,
                 gain_p=0.5, gain_o=0.5, damp=1e-2, max_joint=3.0):
        # Damped-least-squares IK for gripper/base -- position AND orientation.
        model, base_id = self.model, self.base_id
        target = np.asarray(target, float)

        d = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(
            model, d, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home'))

        q_arm = np.array(q_init[:ARM_DOF], float)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        for _ in range(iters):
            d.qpos[:ARM_DOF] = q_arm
            mujoco.mj_forward(model, d)

            err_p = target - np.array(d.xpos[base_id])
            R = np.array(d.xmat[base_id]).reshape(3, 3)
            eR = R_target @ R.T
            err_o = np.array([eR[2, 1] - eR[1, 2],
                              eR[0, 2] - eR[2, 0],
                              eR[1, 0] - eR[0, 1]]) * 0.5

            if np.linalg.norm(err_p) < 1e-5 and np.linalg.norm(err_o) < 1e-6:
                break

            mujoco.mj_jacBody(model, d, jacp, jacr, base_id)
            J = np.vstack([jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]])
            e = np.concatenate([gain_p * err_p, gain_o * err_o])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
            q_arm = np.clip(q_arm + dq, -max_joint, max_joint)

        q_full = d.qpos.copy()
        q_full[:ARM_DOF] = q_arm
        return q_full

    def verify_ik(self):
        # Self-test: drive IK to a known target and report position + orientation.
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, d, home)
        mujoco.mj_forward(self.model, d)

        target = np.array([-0.30, 0.45, 0.30])
        q = self.solve_ik(target, d.qpos.copy())

        d2 = mujoco.MjData(self.model)
        d2.qpos[:] = q
        mujoco.mj_forward(self.model, d2)
        pos_err = np.linalg.norm(np.array(d2.xpos[self.base_id]) - target)
        axis = np.array(d2.xmat[self.base_id]).reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))
        print(f'[IK-CHECK] pos_err={pos_err:.5f} m  tilt={tilt:.2f} deg  '
              f'axis={np.round(axis, 4)}')
        return tilt

    def move(self, q_target, grip, rate=RAPID, viewer=None, ease='quintic',
             estop=None, spawn_hook=None, stop=None):
        # Move to q_target, pausing in place whenever estop() is true.
        data, model = self.data, self.model
        q_start = data.qpos[:ARM_DOF].copy()
        dist = float(np.linalg.norm(np.asarray(q_target)[:ARM_DOF]
                                    - np.asarray(q_start)[:ARM_DOF]))
        floor = FEED_MIN_STEPS if rate <= FEED else RAPID_MIN_STEPS
        steps = max(floor, int(dist / rate / model.opt.timestep))

        # Pace frames by simulated time (~60 Hz), not by move length.
        sync_every = max(1, SYNC_STEPS)
        self._q_ref_prev = np.array(q_start, float)

        for i in range(steps):
            if stop is not None and stop():
                return
            a = (i + 1) / steps
            if ease == 'quintic':
                a = a * a * a * (a * (a * 6.0 - 15.0) + 10.0)
            else:
                a = a * a * (3.0 - 2.0 * a)

            q_ref = a * q_target[:ARM_DOF] + (1 - a) * q_start
            if self.observer is not None:
                # close the outer loop on the estimate
                qd_ref = (q_ref - self._q_ref_prev) / model.opt.timestep
                ctrl_arm = self.controller.command(q_ref, qd_ref,
                                                   self.observer.xhat)
                self.observer.predict(ctrl_arm)
            else:
                ctrl_arm = q_ref
            self._q_ref_prev = q_ref

            data.ctrl[:ARM_DOF] = ctrl_arm
            data.ctrl[self.grip_act] = grip
            mujoco.mj_step(model, data)
            if self.observer is not None:
                self.observer.update(self.encoder.read(data))
            if spawn_hook is not None:
                spawn_hook()
            if viewer is not None and i % sync_every == 0:
                viewer.sync()

            # freeze in place while paused, re-sending the same command
            while estop is not None and estop():
                if stop is not None and stop():
                    return
                data.ctrl[:ARM_DOF] = ctrl_arm
                data.ctrl[self.grip_act] = grip
                if self.observer is not None:
                    # keep tracking while frozen so a resume is not stale
                    self.observer.predict(ctrl_arm)
                mujoco.mj_step(model, data)
                if self.observer is not None:
                    self.observer.update(self.encoder.read(data))
                if spawn_hook is not None:
                    spawn_hook()
                if viewer is not None:
                    viewer.sync()

    def close_gripper(self, steps, viewer=None, estop=None, spawn_hook=None,
                      stop=None):
        # Ramp the fingers shut while holding the present arm pose.
        self._ramp_grip(GRIP_OPEN, GRIP_CLOSED, steps, viewer, estop,
                        spawn_hook, stop)

    def open_gripper(self, steps, viewer=None, estop=None, spawn_hook=None,
                     stop=None):
        # Ramp the fingers open while holding the present arm pose.
        self._ramp_grip(GRIP_CLOSED, GRIP_OPEN, steps, viewer, estop,
                        spawn_hook, stop)

    def _ramp_grip(self, g_from, g_to, steps, viewer, estop=None,
                   spawn_hook=None, stop=None):
        data, model = self.data, self.model
        q_hold = data.qpos[:ARM_DOF].copy()
        for i in range(steps):
            if stop is not None and stop():
                return
            a = (i + 1) / steps
            grip = g_from + a * (g_to - g_from)
            data.ctrl[:ARM_DOF] = q_hold
            data.ctrl[self.grip_act] = grip
            if self.observer is not None:
                self.observer.predict(q_hold)
            mujoco.mj_step(model, data)
            if self.observer is not None:
                self.observer.update(self.encoder.read(data))
            if spawn_hook is not None:
                spawn_hook()
            if viewer is not None and i % SYNC_STEPS == 0:
                viewer.sync()

            while estop is not None and estop():
                if stop is not None and stop():
                    return
                data.ctrl[:ARM_DOF] = q_hold
                data.ctrl[self.grip_act] = grip
                if self.observer is not None:
                    self.observer.predict(q_hold)
                mujoco.mj_step(model, data)
                if self.observer is not None:
                    self.observer.update(self.encoder.read(data))
                if spawn_hook is not None:
                    spawn_hook()
                if viewer is not None:
                    viewer.sync()

    def hold(self, grip):
        # Command the current pose and grip without advancing a ramp.
        data = self.data
        data.ctrl[:ARM_DOF] = data.qpos[:ARM_DOF]
        data.ctrl[self.grip_act] = grip

    def settle(self, steps, grip=GRIP_OPEN):
        # Step in place with the current pose commanded, letting the scene rest.
        data, model = self.data, self.model
        q = data.qpos[:ARM_DOF].copy()
        for _ in range(steps):
            data.ctrl[:ARM_DOF] = q
            data.ctrl[self.grip_act] = grip
            mujoco.mj_step(model, data)
