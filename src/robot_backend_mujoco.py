"""MuJoCo implementation of the RobotBackend I/O seam.

This module is the SIMULATION side of the "who owns the clock" boundary. It
wraps `MjData`: joint targets go to `data.ctrl`, a control cycle is `mj_step`,
and the gripper force comes from the actuator array.

The robot layer never imports this module. A simulation entry point constructs
it and injects it into `Arm`; a hardware entry point injects a `URBackend`
(RTDE `servoj`/`movej`, Modbus for the 2F-85) instead.

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
from src.config import GRIP_STROKE_PER_FINGER_MM


# One scratch MjData per model, so the gravity evaluation does not mutate the
# live state and is not reallocated on every control cycle.
_GRAV_SCRATCH = {}


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

    def __init__(self, model, data, n=6):
        import mujoco
        self._mj = mujoco
        self.model = model
        self.data = data
        self._n = int(n)
        self._dt = float(model.opt.timestep)
        # The gripper is the last actuator in the composed model -- the same
        # assumption `Arm.grip_act` made, kept here instead so the robot layer
        # does not encode it.
        self._grip_act = model.nu - 1
        # Finger travel in metres. The hardware speaks a 0..255 command space; the
        # simulated RG2's actuator is a real slide joint. Conversion lives here so
        # the actuator keeps gear=1 and a physical stiffness.
        self._grip_stroke = GRIP_STROKE_PER_FINGER_MM / 1000.0

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
        # 0..255 (0 = open, as on the hardware) onto 0..stroke in metres.
        self.data.ctrl[self._grip_act] = (float(grip) / 255.0) * self._grip_stroke
        gravity_feedforward(self.model, self.data, self._n)

    def step(self):
        self._mj.mj_step(self.model, self.data)

    # ---- gripper readback ----
    def gripper_state(self):
        # The jaw registers, from the two finger driver joints and the gripper
        # actuator. A URBackend returns the same trio from the RG2's XML-RPC
        # width, which is what lets Arm.grasp_state be identical on both.
        #
        # q is reported as a 0..1 FRACTION of stroke, not metres, so thresholds
        # like GRASP_CLOSED_Q keep their meaning across gripper models.
        q = self._finger_mean('q')
        q = q / self._grip_stroke if self._grip_stroke else float('nan')
        return {
            'q': q,
            'v': self._finger_mean('v') / self._grip_stroke if self._grip_stroke else float('nan'),
            'force': float(self.data.actuator_force[self._grip_act]),
        }

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
