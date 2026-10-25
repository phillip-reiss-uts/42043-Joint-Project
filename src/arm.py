# Arm motion: IK, trajectory execution, gripper, and view repositioning.
#
# HARDWARE BOUNDARY: this module is the robot. It knows about joints, the
# gripper and the viewer, and it knows NOTHING about the object pool, the cube
# driver or the demonstration. It has no reference to any object, and no code
# path here reads or writes object state.
#
# The `on_step` callback replaces the old `viewer=` argument so that this
# module no longer imports the overlay: on hardware there is no viewer, and the
# trajectory executor must compile without one. `main.py` passes viewer.sync.
#
# TWO SEAMS, NO MUJOCO
# --------------------
# This module contains no `import mujoco`. Everything simulator-specific is
# behind a provider, and both providers are REQUIRED constructor arguments:
#
#   src/kinematics.py     FK, Jacobian, M(q), g(q)   -- geometry
#   src/robot_backend.py  stepping, joint I/O, gripper readback -- the clock
#
# The implementations live with their own layer and are never imported here:
#
#   simulation   src/kinematics_mujoco.py  MujocoKinematics
#                src/robot_backend_mujoco.py  MujocoBackend
#   hardware     URKinematics / URBackend, talking to the controller instead
#
# The entry point decides which pair to build, so this module does not change
# between simulation and hardware. `model`/`data` are still accepted because
# the simulation implementations are constructed from them; nothing here
# dereferences them for control.

import numpy as np

from src.config import (
    ARM_DOF, GRIP_OPEN, GRIP_CLOSED, R_DOWN,
    RAPID, FEED, RAPID_MIN_STEPS, FEED_MIN_STEPS,
    SYNC_STEPS,
    GRASP_DRIVER_JOINTS, GRASP_CLOSED_Q, GRASP_FORCE_MIN, GRASP_VEL_TOL,
)
class Arm:
    def __init__(self, model=None, data=None, kinematics=None, backend=None):
        # Both providers are required: `kinematics` supplies FK/Jacobian/M/g,
        # `backend` supplies the clock and joint I/O. There is deliberately no
        # MuJoCo default -- choosing an implementation is the entry point's
        # job, and that is what keeps this module free of the simulator.
        #
        # `model`/`data` are optional and kept only for callers that still pass
        # them; nothing here reads them. On hardware there is no MuJoCo model,
        # so the front-end passes neither.
        if kinematics is None or backend is None:
            raise ValueError(
                'Arm needs both a Kinematics and a RobotBackend provider; '
                'construct them at the entry point (simulation: '
                'src.kinematics_mujoco + src.robot_backend_mujoco)')
        self.model = model
        self.data = data
        # Robot I/O (stepping, joint read/write, gripper readback). Injected so
        # this module does not encode "the gripper is the last actuator" or
        # "a control cycle is mj_step"; both are simulation facts.
        self.backend = backend
        self.base_id = self.backend.tool_body_id
        # Which actuator index the gripper occupies, when the backend has one.
        # On hardware the gripper is a separate device reached over XML-RPC, so
        # this is None.
        self.grip_act = getattr(self.backend, 'grip_actuator', None)

        # Last commanded joint vector, for the observer's predict step.
        self._last_cmd = self.backend.joint_positions()

        # Body currently coupled to the jaws by the parent attach_object().
        # None means the jaws are empty as far as the sim coupling is concerned.
        self._held_body = None

        # Robot geometry (FK, Jacobian, mass/gravity). Injected so the control
        # layer does not reach into the simulator directly.
        self.kin = kinematics

        # Optional closed loop on an estimated state (see controller.py).
        self.observer = None
        self.controller = None
        self.encoder = None

        # Diagnostic sink for observe(). Set by main to the overlay; None on
        # hardware. Typed only as "something with set_info", so this module
        # still does not depend on the overlay layer.
        self.viewer = None

        # Finger driver joints, for reading the jaw opening back. The RG2
        # reports jaw WIDTH in millimetres and the backend scales it into the
        # same units. The backend resolves the addresses, since they are a
        # model detail.
        self.finger_q = []
        self.finger_v = []
        for nm in GRASP_DRIVER_JOINTS:
            adr = self.backend.joint_addresses(nm)
            if adr is not None:
                self.finger_q.append(adr[0])
                self.finger_v.append(adr[1])

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
        u = np.array(self._last_cmd, float)
        self.observer.predict(u)
        self.observer.update(self.encoder.read(self.backend.joint_positions()))
        # Push to the overlay if one is attached, via duck typing rather than a
        # direct import, so this module has no overlay dependency.
        ov = getattr(self.viewer, 'overlay', None)
        if ov is not None:
            ov.set_info(
                'observer',
                'obs err %.5f' % float(np.linalg.norm(
                    self.observer.xhat[:ARM_DOF]
                    - self.backend.joint_positions())),
                'v err %.5f' % float(np.linalg.norm(
                    self.observer.xhat[ARM_DOF:]
                    - self.backend.joint_velocities())))

    def _command_step(self, q_cmd, grip):
        # One control cycle: command, then advance. The two are separate so a
        # hardware backend can hold a command across a cycle, and so the arm
        # never has to know whether ticking blocks (controller) or not (sim).
        self._last_cmd = np.array(q_cmd, float)
        self.backend.command(q_cmd, grip)
        self.backend.step()
        if self.observer is not None:
            self.observer.update(self.encoder.read(self.backend.joint_positions()))

    # IK
    def solve_ik(self, target, q_init, R_target=R_DOWN, iters=600,
                 gain_p=0.5, gain_o=0.5, damp=1e-2, max_joint=3.0):
        # Damped-least-squares IK for gripper/base -- position AND orientation.
        #
        # The solver is pure numpy: FK and the Jacobian come from the injected
        # Kinematics provider. On hardware that provider asks the UR3e
        # controller for the same two quantities, so this loop is unchanged.
        target = np.asarray(target, float)
        q_arm = np.array(q_init[:ARM_DOF], float)

        # `pose_and_jacobian` when available: one FK pass per iteration instead
        # of two. Falls back to the narrow interface otherwise.
        fast = getattr(self.kin, 'pose_and_jacobian', None)

        for _ in range(iters):
            if fast is not None:
                p, R, J = fast(q_arm)
            else:
                p, R = self.kin.pose(q_arm)
                J = self.kin.jacobian(q_arm)

            err_p = target - p
            eR = R_target @ R.T
            err_o = np.array([eR[2, 1] - eR[1, 2],
                              eR[0, 2] - eR[2, 0],
                              eR[1, 0] - eR[0, 1]]) * 0.5

            if np.linalg.norm(err_p) < 1e-5 and np.linalg.norm(err_o) < 1e-6:
                break

            e = np.concatenate([gain_p * err_p, gain_o * err_o])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
            q_arm = np.clip(q_arm + dq, -max_joint, max_joint)

        # Carry the non-arm joints (gripper fingers) through untouched, as
        # before: callers pass the full qpos and expect it back.
        q_full = np.array(q_init, float).copy()
        q_full[:ARM_DOF] = q_arm
        return q_full

    def verify_ik(self):
        # Self-test: drive IK to a known target and report position + orientation.
        # Uses the injected kinematics and backend, so the check exercises the
        # same providers the solver does -- no direct MuJoCo state.
        target = np.array([-0.30, 0.45, 0.30])
        q = self.solve_ik(target, self.backend.joint_positions())

        p, R = self.kin.pose(q[:ARM_DOF])
        pos_err = np.linalg.norm(p - target)
        axis = R[:, 2]
        tilt = np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))
        print(f'[IK-CHECK] pos_err={pos_err:.5f} m  tilt={tilt:.2f} deg  '
              f'axis={np.round(axis, 4)}')
        return tilt

    def move(self, q_target, grip, rate=RAPID, on_step=None, ease='quintic',
             estop=None, stop=None, step_hook=None, viewer=None):
        # Move to q_target, pausing in place whenever estop() is true.
        #
        # on_step is called once per physics step with no arguments. main passes
        # viewer.sync; on hardware it is None or a status/logging callback. The
        # arm does not care what it does.
        # `viewer` is accepted for backwards compatibility and folded into
        # on_step, so no existing call site silently stops rendering.
        if viewer is not None and on_step is None:
            on_step = viewer.sync
        q_start = self.backend.joint_positions()
        # Take the SHORTEST way round. IK returns an angle tuple, not a path, so a
        # target of -3.0 where the arm sits at +3.0 is the same pose but a 6 rad
        # sweep: the wrist and base visibly rotate the long way round, which is
        # what "the camera swings all the way around" was. Every joint here is
        # revolute, so shifting by whole turns is free and lands on the same pose.
        q_target = np.array(q_target, float).copy()
        n = min(ARM_DOF, len(q_target))
        q_target[:n] = q_start[:n] + (
            (q_target[:n] - q_start[:n] + np.pi) % (2.0 * np.pi) - np.pi)
        dist = float(np.linalg.norm(q_target[:ARM_DOF] - q_start[:ARM_DOF]))
        floor = FEED_MIN_STEPS if rate <= FEED else RAPID_MIN_STEPS
        steps = max(floor, int(dist / rate / self.backend.dt))

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
                qd_ref = (q_ref - self._q_ref_prev) / self.backend.dt
                ctrl_arm = self.controller.command(q_ref, qd_ref,
                                                   self.observer.xhat)
                self.observer.predict(ctrl_arm)
            else:
                ctrl_arm = q_ref
            self._q_ref_prev = q_ref

            # before the step, so whatever the hook sets applies to this step
            if step_hook is not None:
                step_hook()

            self._command_step(ctrl_arm, grip)
            if on_step is not None and i % sync_every == 0:
                on_step()

            # freeze in place while paused, re-sending the same command
            while estop is not None and estop():
                if stop is not None and stop():
                    return
                if self.observer is not None:
                    # keep tracking while frozen so a resume is not stale
                    self.observer.predict(ctrl_arm)
                self._command_step(ctrl_arm, grip)
                if on_step is not None:
                    on_step()

    def close_gripper(self, steps, viewer=None, estop=None, stop=None,
                      on_step=None):
        # Ramp the fingers shut while holding the present arm pose.
        self._ramp_grip(GRIP_OPEN, GRIP_CLOSED, steps, viewer, estop,
                        stop, on_step)

    def open_gripper(self, steps, viewer=None, estop=None, stop=None,
                     on_step=None):
        # Ramps open, so whatever the jaws were carrying is released first.
        # Detaching here rather than at each call site means every release path
        # -- pick, place, demo, abort -- uncouples automatically.
        self.detach_object()
        self._ramp_grip(GRIP_CLOSED, GRIP_OPEN, steps, viewer, estop,
                        stop, on_step)

    def attach_object(self, body_name):
        # Couple an object to the jaws for the carry. Simulation only: the
        # hardware backend has no such method and the real gripper needs none.
        # A no-op on hardware rather than an error, so the pick flow stays one
        # sequence on both.
        fn = getattr(self.backend, 'attach_object', None)
        if fn is None:
            return False
        ok = fn(body_name)
        self._held_body = body_name if ok else None
        return ok

    def detach_object(self):
        if self._held_body is None:
            return False
        fn = getattr(self.backend, 'detach_object', None)
        self._held_body = None
        return bool(fn()) if fn is not None else False

    def _ramp_grip(self, g_from, g_to, steps, viewer=None, estop=None,
                   stop=None, on_step=None):
        # `viewer` is accepted for backwards compatibility with existing call
        # sites and folded into on_step, so nothing silently stops rendering.
        if viewer is not None and on_step is None:
            on_step = viewer.sync
        q_hold = self.backend.joint_positions()
        for i in range(steps):
            if stop is not None and stop():
                return
            a = (i + 1) / steps
            grip = g_from + a * (g_to - g_from)
            if self.observer is not None:
                self.observer.predict(q_hold)
            self._command_step(q_hold, grip)
            if on_step is not None and i % SYNC_STEPS == 0:
                on_step()

            while estop is not None and estop():
                if stop is not None and stop():
                    return
                if self.observer is not None:
                    self.observer.predict(q_hold)
                self._command_step(q_hold, grip)
                if on_step is not None:
                    on_step()

    def grasp_state(self):
        # Are the jaws actually holding something? Reads only the gripper's own
        # position, velocity and force, so nothing here depends on knowing where
        # the object is.
        #
        # The backend supplies all three, and they do NOT mean the same thing on
        # both paths: the simulator reads its finger joints and actuator, while
        # a URBackend has RG2 jaw width for q and v and reports NaN for force.
        # NaN is what makes the verdict 'unknown' on hardware, which is the
        # open-loop behaviour chosen deliberately. There is deliberately no read of
        # `data` here, so this one verdict function runs on both.
        #
        # Measured on this model: closing on nothing drives the driver joints to
        # 0.77 with the actuator force at -0.13, the linkage bottoming out.
        # Closing on the 60 mm cube stalls them at 0.25 with the force saturated
        # at +5.0. Three signals agree, so the verdict is taken from all of them
        # rather than trusting the position alone, which cannot separate
        # "gripped the cube" from "stopped against the table".
        g = self.backend.gripper_state()
        q = float(g['q'])
        v = float(g['v'])
        f = float(g['force'])
        if not np.isfinite(q):
            return {'state': 'unknown', 'held': False, 'q': q,
                    'force': f, 'vel': v}

        if q > GRASP_CLOSED_Q:
            state = 'empty'                 # ran to the stop, nothing between
        elif abs(v) > GRASP_VEL_TOL:
            state = 'moving'                # still travelling, read too early
        elif f < GRASP_FORCE_MIN:
            state = 'touching'              # blocked, but not gripping
        else:
            state = 'held'
        return {'state': state, 'held': state == 'held', 'q': q,
                'force': f, 'vel': v}

    def hold(self, grip, q=None):
        # Command a fixed pose and grip without advancing a ramp.
        #
        # `q` MUST be latched by the caller before the loop starts. Passing None
        # re-reads the current position every call, which makes this an
        # integrator rather than a hold: gravity moves the arm slightly, that
        # displaced position becomes the next target, and the arm walks away.
        # Measured over 2000 steps holding a pose, with gravity compensation
        # active in both cases:
        #     feedback target (q=None):  0.214 m of drift
        #     latched target:            0.012 m of drift
        # It is slow enough to look like settling, which is why it went
        # unnoticed; over a WAIT period it reaches ~150 mm at the tool.
        #
        # The q=None form is kept for one-shot "hold where you are" uses, but it
        # must not be called in a loop.
        if q is None:
            q = self.backend.joint_positions()
        self.backend.command(q, grip)

    def hold_pose(self):
        # Latch point for a loop: capture once, then pass to hold() each step.
        return self.backend.joint_positions()

    def settle(self, steps, grip=GRIP_OPEN):
        # Step in place with the current pose commanded, letting the scene rest.
        q = self.backend.joint_positions()
        for _ in range(steps):
            self.backend.command(q, grip)
            self.backend.step()
