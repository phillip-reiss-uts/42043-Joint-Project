"""MuJoCo implementation of the RobotBackend I/O seam.

This module is the SIMULATION side of the "who owns the clock" boundary. It
wraps `MjData`: joint targets go to `data.ctrl`, a control cycle is `mj_step`,
and the gripper force comes from the actuator array.

The robot layer never imports this module. A simulation entry point constructs
it and injects it into `Arm`; a hardware entry point injects a `URBackend`
(RTDE `servoj`/`movej`, XML-RPC for the RG2) instead.

GRAVITY COMPENSATION BELONGS HERE, ONCE
---------------------------------------
A MuJoCo `<position>` servo produces `kp*(ctrl - q) - kv*qvel` and nothing at
zero error, so it cannot hold the arm against its own weight. A UR3e controller
does that internally. The job of this backend is to make the simulated
controller behave like the real one, so it feeds `g(q)` forward -- and the model
must NOT also compensate (`body.gravcomp`), or the arm is pushed up twice.

On hardware the UR controller performs the compensation itself, so `URBackend`
must not send a second one. That is the contract the two implementations share:
**each backend supplies whatever gravity compensation its controller needs,
exactly once.**
"""

import numpy as np

from src.robot_backend import RobotBackend
from src.config import (GRIP_FINGER_OPEN_ANGLE, GRIP_FINGER_SHUT_ANGLE,
                        GRIP_LEVER_MM, UR_CYCLE_DT)


# One scratch MjData per model, so the gravity evaluation does not mutate the
# live state and is not reallocated on every control cycle.
_GRAV_SCRATCH = {}

# A grip command at or below this fraction of the 0..255 stroke counts as "fully
# open", and opening releases whatever the jaws were carrying.
#
# 0.002 is half a unit: loose enough to catch GRIP_OPEN (0.0) and any float
# round-trip through a ramp, tight enough that no genuine partial close trips it.
GRIP_RELEASE_FRAC = 0.002


def gravity_feedforward(model, data, n=6):
    # Write g(q) onto data.qfrc_applied[:n].
    #
    # WHY. Without it a commanded hold accelerates: gravity acts, the position
    # servo responds only after the error builds, and the arm oscillates and
    # walks off (measured: 1.79 rad of drift holding one pose for 3000 steps).
    #
    # ONLY g(q), NOT THE FULL qfrc_bias. qfrc_bias is C(q,qdot) + g(q), and
    # feeding the Coriolis term back is circular: it depends on the very
    # velocity the servo is trying to null, so it pumps energy in and the arm
    # never settles. Evaluating the bias with qvel forced to zero leaves exactly
    # g(q), which is what a static hold needs.
    #
    # The evaluation runs on a scratch MjData so the live state is never
    # mutated -- writing to model.opt.gravity to isolate the term corrupts the
    # run (it left the arm pinned at its start pose).
    #
    # This is the ONE implementation. `pick_place` uses it for its settle loops
    # too, rather than keeping a second copy that could drift out of step.
    import mujoco
    scratch = _GRAV_SCRATCH.get(id(model))
    if scratch is None:
        scratch = _GRAV_SCRATCH[id(model)] = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)
    data.qfrc_applied[:n] = np.asarray(scratch.qfrc_bias[:n], float)


class MujocoBackend(RobotBackend):
    # Simulation backend: wraps MjData, reproducing the previous behaviour
    # exactly. Every method below replaces a direct call that used to live in
    # arm.py / servo.py.

    def __init__(self, model, data, n=6, control_dt=UR_CYCLE_DT):
        import mujoco
        self._mj = mujoco
        self.model = model
        self.data = data
        self._n = int(n)
        # One control cycle is the hardware servoj period, not the physics
        # timestep. The plant still runs at model.opt.timestep inside it.
        self._physics_dt = float(model.opt.timestep)
        self._substeps = max(1, int(round(float(control_dt)
                                          / self._physics_dt)))
        self._dt = self._substeps * self._physics_dt
        # The gripper is the last actuator in the composed model -- the same
        # assumption `Arm.grip_act` made, kept here instead so the robot layer
        # does not encode it.
        self._grip_act = model.nu - 1
        # The official RG2 is driven through its four-bar linkage: one knuckle
        # angle sweeps -0.35692 rad (open) to +0.785398 rad (shut), and the other
        # five joints follow through equality constraints. The hardware speaks a
        # 0..255 command space with 0 = OPEN, which is the opposite direction, so
        # the inversion lives here rather than in the model.
        self._grip_open = float(GRIP_FINGER_OPEN_ANGLE)
        self._grip_shut = float(GRIP_FINGER_SHUT_ANGLE)
        self._grip_span = self._grip_shut - self._grip_open
        self._grip_joint = 'gripper/finger_joint'
        # Body being carried by the jaws (see attach_object), or None.
        self._carry = None

    # ---- state ----
    def joint_positions(self):
        return np.array(self.data.qpos[:self._n], float)

    def joint_velocities(self):
        return np.array(self.data.qvel[:self._n], float)

    def time(self):
        return float(self.data.time)

    @property
    def dt(self):
        return self._dt

    # ---- commands ----
    def command(self, q_arm, grip):
        self.data.ctrl[:self._n] = np.asarray(q_arm, float)
        # 0..255 (0 = open, as on the hardware) onto the linkage knuckle angle.
        f = float(grip) / 255.0                      # 0 = open, 1 = shut
        self.data.ctrl[self._grip_act] = self._grip_open + f * self._grip_span
        gravity_feedforward(self.model, self.data, self._n)
        # OPENING THE JAWS RELEASES THE OBJECT.
        #
        # _carry_object re-writes the cube's pose every step while a carry is
        # live, so any path that commands the jaws open WITHOUT going through
        # Arm.open_gripper used to leave the cube welded to the tool as the
        # fingers withdrew: it then hung ~230 mm below the gripper, directly in
        # front of the wrist camera, and the next look found a cube where the
        # table should be. Arm.open_gripper detaches explicitly, but
        # arm.move(..., GRIP_OPEN), arm.hold(GRIP_OPEN) and the servo's own
        # per-step command all come through here, and BOTH the failed-pick
        # re-home and clear_table's recovery search are plain GRIP_OPEN moves.
        #
        # Enforcing it at the single funnel is deliberate: the coupling owns the
        # release rule, so a new GRIP_OPEN call site cannot reintroduce this.
        if self._carry is not None and f <= GRIP_RELEASE_FRAC:
            self.detach_object()

    def step(self):
        # One hardware control cycle: several physics steps, one 125 Hz tick.
        for _ in range(self._substeps):
            self._carry_object()
            self._mj.mj_step(self.model, self.data)
        self._carry_object()

    # ---- grasp coupling (simulation only) ----
    def attach_object(self, body_name):
        # Carry `body_name` with the tool, in the relative pose it has right now.
        #
        # The simulated pads are rigid where the real ones are rubber, so the cube
        # creeps down the pad faces under its own weight and falls out of a
        # top-flush grip. A weld EQUALITY between a 90 g free body and the arm
        # chain is numerically unstable (the cube's inertia is tiny, so the
        # constraint torque flings it), so the body is carried kinematically
        # instead: its free joint is rewritten to follow the tool every step.
        #
        # This is a deliberate simplification, not a contact model: once the jaws
        # have closed on the object it moves with them, which is what the real
        # gripper does.
        import mujoco
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
        if b < 0:
            return False
        j = int(self.model.body_jntadr[b])
        R_t = np.array(self.data.xmat[int(self.tool_body_id)], float).reshape(3, 3)
        p_t = np.array(self.data.xpos[int(self.tool_body_id)], float)
        R_b = np.array(self.data.xmat[b], float).reshape(3, 3)
        p_b = np.array(self.data.xpos[b], float)
        self._carry = {
            'qadr': int(self.model.jnt_qposadr[j]),
            'dadr': int(self.model.jnt_dofadr[j]),
            'rel_R': R_t.T @ R_b,      # object orientation in the tool frame
            'rel_p': R_t.T @ (p_b - p_t),
        }
        return True

    def detach_object(self):
        # Stop carrying: the body keeps whatever pose it was left in.
        self._carry = None
        return True

    def _carry_object(self):
        if self._carry is None:
            return
        c = self._carry
        R_t = np.array(self.data.xmat[int(self.tool_body_id)], float).reshape(3, 3)
        p_t = np.array(self.data.xpos[int(self.tool_body_id)], float)
        R_b = R_t @ c['rel_R']
        p_b = p_t + R_t @ c['rel_p']
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, R_b.ravel())
        self.data.qpos[c['qadr']:c['qadr'] + 3] = p_b
        self.data.qpos[c['qadr'] + 3:c['qadr'] + 7] = quat
        self.data.qvel[c['dadr']:c['dadr'] + 6] = 0.0

    # ---- gripper readback ----
    def gripper_state(self):
        # The jaw registers, from the gripper's own DOF and its actuator. A
        # URBackend returns the same trio from the RG2's XML-RPC width, which is
        # what lets Arm.grasp_state be identical on both. 'detected' is None
        # here: the simulator has no contact sensor.
        #
        # q is reported as a 0..1 FRACTION of stroke, not metres, so thresholds
        # like GRASP_CLOSED_Q keep their meaning across gripper models.
        q = (self._grip_pos('q') - self._grip_open) / self._grip_span
        v = self._grip_pos('v') / self._grip_span
        return {
            'q': q,
            'v': v,
            # The actuator sits on the finger HINGE, so actuator_force is a TORQUE
            # (N·m). The hardware path reports the force AT THE PADS (N), which is
            # what GRASP_FORCE_MIN is calibrated against. Convert through the lever
            # arm so the same verdict logic works on both.
            'force': float(self.data.actuator_force[self._grip_act])
            / (GRIP_LEVER_MM / 1000.0),
            # Contact sensing does not exist in the simulator; None means "no
            # opinion", which is what the hardware field also reports when the
            # URCap will not answer.
            'detected': None,
            # No millimetre readback in the simulator; the settle loop is
            # hardware-only and never asks for it.
            'width_mm': None,
        }

    def _grip_pos(self, which):
        # Position or velocity of the gripper's driven DOF, or NaN if the model
        # has no gripper.
        #
        # The official RG2 exposes ONE prismatic joint, finger_width. The older
        # proxy exposed a pair of driver joints, so fall back to their mean to keep
        # this working for either model.
        import mujoco
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self._grip_joint)
        if j >= 0:
            if which == 'q':
                return float(self.data.qpos[self.model.jnt_qposadr[j]])
            return float(self.data.qvel[self.model.jnt_dofadr[j]])
        return self._finger_mean(which)

    def _finger_mean(self, which):
        # Mean position or velocity of the two driver joints, or NaN if they
        # are not in this model.
        import mujoco
        vals = []
        for name in ('gripper/right_driver_joint', 'gripper/left_driver_joint'):
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                continue
            if which == 'q':
                vals.append(float(self.data.qpos[self.model.jnt_qposadr[j]]))
            else:
                vals.append(float(self.data.qvel[self.model.jnt_dofadr[j]]))
        return float(np.mean(vals)) if vals else float('nan')

    def joint_addresses(self, name):
        import mujoco
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            return None
        return int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])

    @property
    def grip_actuator(self):
        return self._grip_act

    @property
    def tool_body_id(self):
        # The body whose pose the robot cares about (the gripper flange).
        import mujoco
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    'gripper/base'))
