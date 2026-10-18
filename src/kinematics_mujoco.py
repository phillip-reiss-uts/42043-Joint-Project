"""MuJoCo implementation of the Kinematics provider.

This module is the SIMULATION side of the geometry seam. It contains every
direct MuJoCo geometry call the robot layer used to make, wrapped so that
`src/kinematics.py` can define the interface without importing a simulator.

The robot layer never imports this module. A simulation entry point constructs
it and injects it into `Arm`; a hardware entry point injects a `URKinematics`
instead and nothing else changes.

WHY THE SEPARATION EXISTS
-------------------------
`Kinematics` (src/kinematics.py) is the question "given joint angles, where is
the tool and how does it move?". A real UR3e controller answers that over RTDE.
MuJoCo answers it by running `mj_kinematics`. Those are different
implementations of the same interface, so they live in different modules:
the portable interface is shared, this one is simulation-only.
"""

import numpy as np

from src.kinematics import Kinematics


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
