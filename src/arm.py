"""Arm motion: IK, trajectory execution, gripper, and view repositioning.

Owns the model/data pair it drives. The viewer stays a method parameter rather
than instance state, because passing it to launch_passive rebinds it and a
stored reference would go stale.
"""

import mujoco
import numpy as np

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, R_DOWN,
    RAPID, FEED, RAPID_MIN_STEPS, FEED_MIN_STEPS,
    SYNC_STEPS,
    # LEFTOVER: SYNC_DT is documentation for where SYNC_STEPS comes from; the
    # code only ever uses SYNC_STEPS.
    # SYNC_DT,
    # TABLE_HEIGHT was only used by the commented-out peek_target.
    # TABLE_HEIGHT,
)


class Arm:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
        self.grip_act = model.nu - 1          # fingers_actuator

    # ------------------------------------------------------------------ IK ---
    def solve_ik(self, target, q_init, R_target=R_DOWN, iters=600,
                 gain_p=0.5, gain_o=0.5, damp=1e-2, max_joint=3.0):
        """Damped-least-squares IK for gripper/base -- position AND orientation.

        mink's QP finds *a* solution, not the nearest one: for this target it
        winds wrist_1 to 4.29 rad and wrist_3 to -4.37 rad, so the arm takes a
        full turn (~6.8 rad of travel) before reaching the pose. DLS from the
        current configuration stays local, giving ~0.4 rad for the same target.

        R_target pins the tool orientation. The default R_DOWN (approach axis
        along world -z) leaves rotation about that axis unconstrained -- harmless
        for an approach or grasp, but for a pure lateral peek the solver picks an
        arbitrary wrist_3 and spends ~1.77 rad turning instead of translating.
        Pass the current orientation to suppress that.

        Operates on the arm joints only; the object freejoint keeps its keyframe
        value. Returns a full-length qpos.
        """
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
        """Self-test: drive IK to a known target and report position + orientation.

        Position-only IK reaches the right point with the tool tilted (30 deg at
        the grasp pose), which leaves the fingers above the object closing along
        a skewed axis. This is the check that catches that class of regression.
        """
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
             estop=None):
        """Move to q_target, pausing in place whenever estop() is true.

        estop is polled every step: while it returns True the arm re-commands the
        current interpolated pose and does not advance the trajectory, so an
        e-stop freezes the arm exactly where it is and resume continues the same
        move from there (Option A, no re-plan).
        """
        data, model = self.data, self.model
        q_start = data.qpos[:ARM_DOF].copy()
        dist = float(np.linalg.norm(np.asarray(q_target)[:ARM_DOF]
                                    - np.asarray(q_start)[:ARM_DOF]))
        floor = FEED_MIN_STEPS if rate <= FEED else RAPID_MIN_STEPS
        steps = max(floor, int(dist / rate / model.opt.timestep))

        # Pace frames by simulated time (~60 Hz), not by move length: a fixed
        # interval either stutters on a short fast hop or burns hundreds of
        # renders on a long slow one. Syncs then total ~(sim duration)/16ms.
        sync_every = max(1, SYNC_STEPS)

        for i in range(steps):
            a = (i + 1) / steps
            if ease == 'quintic':
                a = a * a * a * (a * (a * 6.0 - 15.0) + 10.0)
            else:
                a = a * a * (3.0 - 2.0 * a)

            ctrl_arm = a * q_target[:ARM_DOF] + (1 - a) * q_start
            data.ctrl[:ARM_DOF] = ctrl_arm
            data.ctrl[self.grip_act] = grip
            mujoco.mj_step(model, data)
            if viewer is not None and i % sync_every == 0:
                viewer.sync()

            # freeze in place while paused; keep re-sending the same command so
            # gravity comp doesn't let the arm drift, and keep syncing so the
            # key callback still fires.
            while estop is not None and estop():
                data.ctrl[:ARM_DOF] = ctrl_arm
                data.ctrl[self.grip_act] = grip
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()

    def close_gripper(self, steps, viewer=None, estop=None):
        """Ramp the fingers shut while holding the present arm pose.

        The grip command is interpolated rather than stepped: commanding 0 -> 255
        in one tick drives the fingers into the object and the contact solver
        spikes to ~412 N. Ramping over this phase gives the same final grip at
        ~80 N peak.
        """
        self._ramp_grip(GRIP_OPEN, GRIP_CLOSED, steps, viewer, estop)

    def open_gripper(self, steps, viewer=None, estop=None):
        """Ramp the fingers open while holding the present arm pose."""
        self._ramp_grip(GRIP_CLOSED, GRIP_OPEN, steps, viewer, estop)

    def _ramp_grip(self, g_from, g_to, steps, viewer, estop=None):
        data, model = self.data, self.model
        q_hold = data.qpos[:ARM_DOF].copy()
        for i in range(steps):
            a = (i + 1) / steps
            grip = g_from + a * (g_to - g_from)
            data.ctrl[:ARM_DOF] = q_hold
            data.ctrl[self.grip_act] = grip
            mujoco.mj_step(model, data)
            if viewer is not None and i % SYNC_STEPS == 0:
                viewer.sync()

            while estop is not None and estop():
                data.ctrl[:ARM_DOF] = q_hold
                data.ctrl[self.grip_act] = grip
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()

    def hold(self, grip):
        """Command the current pose and grip without advancing a ramp."""
        data = self.data
        data.ctrl[:ARM_DOF] = data.qpos[:ARM_DOF]
        data.ctrl[self.grip_act] = grip

    def settle(self, steps, grip=GRIP_OPEN):
        """Step in place with the current pose commanded, letting the scene rest."""
        data, model = self.data, self.model
        q = data.qpos[:ARM_DOF].copy()
        for _ in range(steps):
            data.ctrl[:ARM_DOF] = q
            data.ctrl[self.grip_act] = grip
            mujoco.mj_step(model, data)

    # ---------------------------------------------------------------- view ---
    # LEFTOVER: peek_target is not called anywhere. It was the singleton
    # "nudge the camera sideways" helper; the search inside clear_table now does
    # translate + revolve instead.
    # def peek_target(self, q_home, obj_xyz, dx, dy):
    #     """Joints for a lateral viewing offset, holding the current orientation.
    #
    #     Does NOT move the arm -- returns the joint target so the caller decides
    #     how to get there. Returns q_home unchanged if every offset failed IK.
    #     """
    #     obj_xyz = np.asarray(obj_xyz, float).reshape(-1)
    #     if obj_xyz.size != 3:
    #         return q_home
    #
    #     base_z = float(self.data.xpos[self.base_id][2])
    #
    #     d_hold = mujoco.MjData(self.model)
    #     d_hold.qpos[:] = self.data.qpos
    #     mujoco.mj_forward(self.model, d_hold)
    #     R_hold = np.array(d_hold.xmat[self.base_id]).reshape(3, 3)
    #
    #     for scale in (1.0, 1.6, 2.4):
    #         target = [obj_xyz[0] + dx * scale, obj_xyz[1] + dy * scale, base_z]
    #         q = self.solve_ik(target, q_home, R_target=R_hold)
    #         check = mujoco.MjData(self.model)
    #         check.qpos[:] = q
    #         mujoco.mj_forward(self.model, check)
    #         posed = np.array(check.xpos[self.base_id])
    #         if np.linalg.norm(posed - np.array(target)) < 5e-3:
    #             cam = np.array(check.cam_xpos[
    #                 mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
    #                                   'depth_cam')])
    #             if cam[2] > TABLE_HEIGHT + 0.05:
    #                 return q
    #     return q_home