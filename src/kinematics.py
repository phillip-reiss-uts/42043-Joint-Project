"""Kinematics: the robot's own forward model, as an interface.

WHY THIS EXISTS
---------------
The robot side of this project must run on a real UR3e. Its control logic --
the PBVS law, the EKF, the LQR design, the damped-least-squares IK -- is pure
numpy and already portable. What is NOT portable is where those algorithms get
their *robot geometry* from.

The question is always the same: "given joint angles, where is the tool, and
how does it move?" That question has a different answer on hardware (the UR3e
controller answers it) than in simulation (MuJoCo answers it), so it belongs
behind an interface.

This module defines the interface and supplies the hardware-portable
implementation. It imports no simulator and no device driver. The
implementations live with their own layer:

    src/kinematics.py          this file: the interface, plus AnalyticKinematics
    src/kinematics_mujoco.py   MujocoKinematics -- MuJoCo only
    (hardware)                 URKinematics -- reads FK/Jacobian from the UR3e
                               controller, with src/arm_dynamics.py supplying
                               M(q) and g(q)

`AnalyticKinematics` is shared rather than simulation code because it computes
`M(q)`/`g(q)` from published link data with a recursive Newton-Euler
formulation (src/arm_dynamics.py) -- no simulator involved. It is verified
against MuJoCo to machine precision by tests/test_dynamics.py.

MASS/GRAVITY AND SERVO GAINS
----------------------------
A UR3e controller will not hand you `M(q)`, `g(q)` or its servo stiffness. Both
are therefore modelled rather than queried:

  * `AnalyticKinematics` computes M and g from published link data with
    a recursive Newton-Euler formulation. It is verified to machine precision
    against MuJoCo, so it is a drop-in replacement rather than an
    approximation.

  * `servo_gains()` describes the position servo, which is what `A_cl` in the
    linearisation is built from. MuJoCo's actuator gains are a simulation fact;
    a hardware implementation must supply the controller's own stiffness, and
    where that is unknown it should be identified from a step response. The
    class below flags this rather than silently returning simulation numbers.
"""

from abc import ABC, abstractmethod

import numpy as np


class Kinematics(ABC):
    # The robot geometry the control layer is allowed to ask about.
    #
    # No object state, no scene, no physics stepping. Only: where is the tool,
    # how does it move, and what are the arm's inertial properties.

    @abstractmethod
    def pose(self, q):
        # (position, rotation matrix) of the tool/gripper base for joint config q.
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, q):
        # 6x6 geometric Jacobian of the tool, base frame: [linear; angular].
        raise NotImplementedError

    @abstractmethod
    def mass_matrix(self, q):
        # (n, n) joint-space inertia matrix M(q), including actuator armature.
        raise NotImplementedError

    @abstractmethod
    def gravity_torque(self, q):
        # (n,) joint torque required to hold q against gravity, g(q).
        raise NotImplementedError

    def mass_gravity(self, q):
        # Convenience: both at once. Implementations may override with a single
        # pass over the model, which is why this is a method rather than a
        # caller-side pair.
        return self.mass_matrix(q), self.gravity_torque(q)

    @property
    def dof(self):
        return 6


class AnalyticKinematics(Kinematics):
    # Hardware-portable kinematics: FK and Jacobian from a frame chain, M(q) and
    # g(q) from recursive Newton-Euler. No simulator involved.
    #
    # This is what a UR3e build would use. The dynamics come from published link
    # data (UR URDF + payload), and `servo_gains` is supplied explicitly --
    # there is deliberately no default, because the numbers must come from the
    # real controller rather than from a simulator's actuator table.
    #
    # `frame_provider` must supply, for a given q:
    #     pose(q)          -> (p, R) of the tool
    #     jacobian(q)      -> 6x6
    # and `dyn` is an ArmDynamics instance (src/arm_dynamics.py).

    def __init__(self, frame_provider, dyn, servo_gains):
        self._frames = frame_provider
        self.dyn = dyn
        self._gains = servo_gains
        if servo_gains is None:
            raise ValueError(
                'AnalyticKinematics needs explicit servo gains: the real '
                "controller's stiffness is not knowable from the model alone")

    def pose(self, q):
        return self._frames.pose(q)

    def jacobian(self, q):
        return self._frames.jacobian(q)

    def mass_matrix(self, q):
        return self.dyn.mass_matrix(np.asarray(q, float))

    def gravity_torque(self, q):
        return self.dyn.gravity_torque(np.asarray(q, float))

    def mass_gravity(self, q):
        return self.dyn.mass_gravity(np.asarray(q, float))

    def servo_gains(self):
        return self._gains
