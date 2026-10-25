"""Robot backend: who owns the clock, and how joint state is read and written.

WHY THIS EXISTS
---------------
`src/kinematics.py` moved the robot's *geometry* behind an interface. This
module moves the robot's *I/O* behind one: stepping a control cycle, reading
joint state, and commanding the arm and gripper.

Before the split, `arm.py` and `servo.py` each did:

    data.ctrl[:ARM_DOF] = q_cmd
    data.ctrl[model.nu - 1] = GRIP_OPEN
    mujoco.mj_step(model, data)

That is three hardware assumptions in four lines:

  1. arm commands are a vector of joint positions at one address range;
  2. the gripper is the last actuator in that same vector;
  3. a control cycle is `mj_step`, i.e. the simulation advances time.

On a real UR3e none of those hold. Arm targets go over RTDE as `servoj`/`movej`,
the RG2's jaw is commanded over XML-RPC and reports width, and the clock is the
controller's 125 Hz control loop -- you do not advance it, you participate in it.

This module is the interface only. The implementations live with their layer:

    src/robot_backend_mujoco.py   MujocoBackend -- simulation
    (hardware)                    URBackend -- RTDE + RG2

`Arm` is handed whichever one the entry point constructed, so the robot layer
never imports either.

WHAT THIS DOES NOT SOLVE
------------------------
A real `step()` blocks until the controller's next cycle, and `read()` returns
what the encoders actually say. The simulated backend steps instantly and
returns perfect state. That is the difference an observer exists to cover, and
it is why `ArmObserver` takes encoder noise seriously.
"""

from abc import ABC, abstractmethod


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
        # velocity and actuator force. The simulator can supply all three; the
        # RG2 supplies width (so q and v) and returns NaN for force, which is
        # why Arm.grasp_state reports 'unknown' on hardware.
        raise NotImplementedError

    # ---- model handles ----
    @abstractmethod
    def joint_addresses(self, name):
        # (qpos_index, qvel_index) of a named joint, or None if absent.
        # Addresses are a simulator concept; a hardware backend would return a
        # logical index into its own joint list instead.
        raise NotImplementedError
