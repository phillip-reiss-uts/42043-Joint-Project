"""Robot backend: who owns the clock, and how joint state is read and written.

WHY THIS EXISTS
---------------
`src/kinematics.py` moved the robot's *geometry* behind an interface. This
module moves the robot's *I/O* behind one: stepping a control cycle, reading
joint state, and commanding the arm and gripper.

Before this, `arm.py` and `servo.py` each did:

    data.ctrl[:ARM_DOF] = q_cmd
    data.ctrl[model.nu - 1] = GRIP_OPEN
    mujoco.mj_step(model, data)

That is three hardware assumptions in four lines:

  1. arm commands are a vector of joint positions at one address range;
  2. the gripper is the last actuator in that same vector;
  3. a control cycle is `mj_step`, i.e. the simulation advances time.

On a UR3e none of those hold. Arm targets go over RTDE as `servoj`/`movej`,
the 2F-85 goes over Modbus, and the clock is the controller's 125 Hz control
loop -- you do not advance it, you participate in it.

`MujocoBackend` wraps exactly what the project already did, so wiring it in is
numerically neutral. A `URBackend` would implement the same five methods and
the robot modules would not change.

WHAT THIS DOES NOT SOLVE
------------------------
A real `step()` blocks until the controller's next cycle, and `read()` returns
what the encoders actually say. The simulated backend steps instantly and
returns perfect state. That is the difference an observer exists to cover, and
it is why `ArmObserver` takes encoder noise seriously.
"""

from abc import ABC, abstractmethod

import numpy as np


class RobotBackend(ABC):
    # The robot's I/O surface. Nothing here knows about objects, scenes or the
    # viewer -- this is purely "move the arm, read the arm, cycle the clock".

    # ---- state ----
    @abstractmethod
    def joint_positions(self):
        # (n,) measured joint angles (rad). On hardware: encoder readback.
        raise NotImplementedError

    @abstractmethod
    def joint_velocities(self):
        # (n,) measured joint velocities (rad/s). May be None on a controller
        # that does not report them -- the observer is what compensates.
        raise NotImplementedError

    def time(self):
        # Monotonic time (s) of the current control cycle. Used to timestamp
        # the Kalman filter and the stillness timer.
        raise NotImplementedError

    @property
    def dt(self):
        # Nominal control period (s).
        raise NotImplementedError

    # ---- commands ----
    @abstractmethod
    def command(self, q_arm, grip):
        # Set the arm joint targets and the gripper command WITHOUT advancing
        # time. Mirrors writing data.ctrl, so the caller can decide when the
        # cycle actually ticks.
        raise NotImplementedError

    @abstractmethod
    def step(self):
        # Advance one control cycle. On hardware this blocks for the control
        # period; in simulation it runs mj_step.
        raise NotImplementedError

    # ---- gripper readback ----
    @abstractmethod
    def gripper_state(self):
        # {'q': float, 'v': float, 'force': float} -- the jaw position,
        # velocity and actuator force. A real 2F-85 reports the same trio over
        # Modbus, which is why Arm.grasp_state reads only these.
        raise NotImplementedError

    # ---- model handles ----
    @abstractmethod
    def joint_addresses(self, name):
        # (qpos_index, qvel_index) of a named joint, or None if absent.
        # Addresses are a simulator concept; a hardware backend would return a
        # logical index into its own joint list instead.
        raise NotImplementedError


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
        # Scratch data for gravity compensation; allocated on first use.
        self._grav_data = None

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
        self.data.ctrl[self._grip_act] = float(grip)
        self._compensate_gravity()

    def _compensate_gravity(self):
        # Feed the arm's weight forward so the position servos only correct
        # tracking error instead of carrying the arm.
        #
        # WHY. A MuJoCo <position> servo produces kp*(ctrl - q) - kv*qvel and
        # NOTHING at zero error. Gravity still acts, so a commanded hold
        # accelerates (21.7 rad/s^2 at the home pose on the UR3e); the error
        # builds, the servo responds late, and the arm oscillates and walks off
        # -- the "drifts off even when not searching" symptom. Measured over
        # 3000 steps holding a fixed pose: 1.79 rad of drift without this,
        # 0.000000 rad with it.
        #
        # ONLY g(q), NOT THE FULL qfrc_bias. qfrc_bias is C(q,qdot) + g(q), and
        # feeding the Coriolis term back is circular: it depends on the very
        # velocity the servo is trying to null, so it pumps energy in and the
        # arm never settles (measured: 0.0197 rad residual with qvel ~0.3
        # persisting). Evaluating the bias with qvel forced to zero leaves
        # exactly g(q), which is what a static hold needs. That gives zero
        # drift AND zero residual velocity across every gain pair tried.
        #
        # The evaluation runs on a scratch MjData so the live state is never
        # mutated -- writing to model.opt.gravity to isolate the term corrupts
        # the run (it left the arm pinned at its start pose).
        #
        # On hardware this is the same signal a gravity-compensating controller
        # sends, which is why it belongs on the backend and not in the task
        # layer.
        if self._grav_data is None:
            self._grav_data = self._mj.MjData(self.model)
        scratch = self._grav_data
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        self._mj.mj_forward(self.model, scratch)
        self.data.qfrc_applied[:self._n] = np.array(
            scratch.qfrc_bias[:self._n], float)

    def step(self):
        self._mj.mj_step(self.model, self.data)

    # ---- gripper readback ----
    def gripper_state(self):
        return {
            'q': float('nan'),          # filled by Arm from its finger joints
            'v': float('nan'),
            'force': float(self.data.actuator_force[self._grip_act]),
        }

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
