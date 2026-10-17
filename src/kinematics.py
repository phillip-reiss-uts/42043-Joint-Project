"""Kinematics: the robot's own forward model, injected rather than imported.

WHY THIS EXISTS
---------------
The robot side of this project must run on a real UR3e. Its control logic --
the PBVS law, the EKF, the LQR design, the damped-least-squares IK -- is pure
numpy and already portable. What is NOT portable is where those algorithms get
their *robot geometry* from.

Today they call into MuJoCo directly:

    arm.solve_ik      -> mj_kinematics, mj_comPos, mj_jacBody
    servo.joint_step  -> mj_jacBody
    observer          -> mj_fullM, mj_kinematics, mj_crb/rne
    state_space       -> mj_fullM, mj_kinematics, mj_rne, actuator gains

Every one of those is the same question: "given joint angles, where is the
tool, and how does it move?" That question has a different answer on hardware
(the UR3e controller answers it), so it belongs behind an interface.

This module is that interface. `MujocoKinematics` wraps today's calls so nothing
changes numerically. A `URKinematics` implementation would wrap the real
controller and the robot modules would not change.

DESIGN NOTE ON MASS/GRAVITY
---------------------------
`mass_gravity()` has no clean UR3e equivalent: the controller will not hand you
an inertia matrix. It is part of the interface because the observer and the
linearisation genuinely need it, but a real implementation has to either use a
pinned model or a system-identified one. Marked clearly below so nobody assumes
the port is free.
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


class MujocoKinematics(Kinematics):
    # MuJoCo-backed kinematics: wraps the exact calls the project already made,
    # so wiring this in is numerically neutral.

    def __init__(self, model, tool_body_id, n=6):
        import mujoco
        self._mj = mujoco
        self.model = model
        self.tool = int(tool_body_id)
        self._n = int(n)

        # Scratch data for FK/Jacobian. Reused rather than reallocated: solve_ik
        # calls pose()/jacobian() up to 600 times per solve.
        self._scratch = mujoco.MjData(model)
        self._Mbuf = np.zeros((model.nv, model.nv))
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # GEOMETRY
    def _fk(self, q):
        # Kinematics + comPos only. The full mj_forward costs ~5x more and this
        # runs in the inner loop of the IK solver.
        mj, d = self._mj, self._scratch
        d.qpos[:self._n] = q
        mj.mj_kinematics(self.model, d)
        mj.mj_comPos(self.model, d)
        return d

    def pose(self, q):
        d = self._fk(np.asarray(q, float))
        p = np.array(d.xpos[self.tool], float)
        R = np.array(d.xmat[self.tool], float).reshape(3, 3)
        return p, R

    def jacobian(self, q):
        d = self._fk(np.asarray(q, float))
        self._mj.mj_jacBody(self.model, d, self._jacp, self._jacr, self.tool)
        return np.vstack([self._jacp[:, :self._n], self._jacr[:, :self._n]])

    def pose_and_jacobian(self, q):
        # One FK pass instead of two: pose() and jacobian() both need the same
        # kinematics, and the IK loop calls both every iteration.
        d = self._fk(np.asarray(q, float))
        p = np.array(d.xpos[self.tool], float)
        R = np.array(d.xmat[self.tool], float).reshape(3, 3)
        self._mj.mj_jacBody(self.model, d, self._jacp, self._jacr, self.tool)
        J = np.vstack([self._jacp[:, :self._n], self._jacr[:, :self._n]])
        return p, R, J

    # DYNAMICS
    def _forward_dynamics(self, q):
        mj, d = self._mj, self._scratch
        d.qpos[:self._n] = np.asarray(q, float)
        d.qvel[:] = 0.0
        mj.mj_kinematics(self.model, d)
        mj.mj_comPos(self.model, d)
        mj.mj_crb(self.model, d)
        mj.mj_factorM(self.model, d)
        mj.mj_rne(self.model, d, 0, d.qfrc_bias)
        return d

    def mass_matrix(self, q):
        d = self._forward_dynamics(q)
        self._mj.mj_fullM(self.model, d, self._Mbuf)
        return self._Mbuf[:self._n, :self._n].copy()

    def gravity_torque(self, q):
        d = self._forward_dynamics(q)
        return np.array(d.qfrc_bias[:self._n], float)

    def mass_gravity(self, q):
        # One pass for both, as promised by the base class.
        d = self._forward_dynamics(q)
        self._mj.mj_fullM(self.model, d, self._Mbuf)
        return (self._Mbuf[:self._n, :self._n].copy(),
                np.array(d.qfrc_bias[:self._n], float))

    # SERVO MODEL
    def servo_gains(self):
        # The simulator's position-servo (Kp, Kv). NOTE: this describes the
        # SIMULATED actuator, not a real UR3e servo. The linearisation uses it
        # to form A_cl, so on hardware this needs the real controller's
        # stiffness -- which UR does not publish precisely. Flagged here rather
        # than silently returning sim numbers.
        m = self.model
        Kp = np.array([m.actuator_gainprm[i][0] for i in range(self._n)])
        Kv = np.array([-m.actuator_biasprm[i][2] for i in range(self._n)])
        return Kp, Kv
