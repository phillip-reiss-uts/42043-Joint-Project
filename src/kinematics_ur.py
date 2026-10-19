"""Hardware kinematics: the UR3e implementation of the Kinematics provider.

WHY THIS IS A THIN WRAPPER
--------------------------
`Kinematics` (src/kinematics.py) asks for four things: FK, the Jacobian, M(q)
and g(q). None of them needs a simulator, and this project already has a
hardware-portable implementation of all four:

    src/arm_dynamics.py   a 6R chain with link masses, COMs and inertias.
                          `frames(q)` is forward kinematics; `mass_matrix` and
                          `gravity_torque` are recursive Newton-Euler.

So the hardware provider is `AnalyticKinematics` over a frame chain built from
those links. There is no second model to keep in sync with the simulation, and
`tests/test_ur_kinematics.py` checks the whole chain against MuJoCo to machine
precision -- offline, with no robot.

TOOL FRAME
----------
The chain's last link is wrist_3_link. The frame the robot actually controls is
the gripper base, which sits at a fixed offset and rotation from it (the
gripper's mounting adapter). That transform is a property of the hardware, so it
is a constructor argument, not a constant here.

`tests/test_ur_kinematics.py` derives it from the composed model and checks the
whole chain against MuJoCo, so a mistake in the mount shows up offline.

WHAT MUST COME FROM THE ROBOT
-----------------------------
`servo_gains()` is the one quantity the link data cannot give you: it is a
property of the controller, not the arm. It has no default here, for the same
reason `AnalyticKinematics` has none -- passing the simulated 1000/200 would
silently describe the wrong plant. `tests/test_hardware.py --real` describes how
to identify it from a step response.
"""

import numpy as np

from src.arm_dynamics import rot_from_quat
from src.kinematics import AnalyticKinematics


class ChainFrames:
    # FK and the geometric Jacobian of a serial chain, from `ArmDynamics.frames`.
    #
    # The tool frame is the last link's frame composed with a fixed mount
    # transform (position in that frame, orientation as a quaternion), which is
    # where the gripper base actually is.

    def __init__(self, dyn, tool_pos=(0.0, 0.0, 0.0),
                 tool_quat=(1.0, 0.0, 0.0, 0.0)):
        self.dyn = dyn
        self.tool_pos = np.asarray(tool_pos, float)
        self.tool_rot = rot_from_quat(tool_quat)

    def _tool(self, last):
        # Tool pose from the last link's (R, p).
        p = np.asarray(last['p'], float) + last['R'] @ self.tool_pos
        return p, last['R'] @ self.tool_rot

    def pose(self, q):
        return self._tool(self.dyn.frames(np.asarray(q, float))[-1])

    def jacobian(self, q):
        # Geometric Jacobian, base frame: J = [z_i x (p_tool - p_i); z_i].
        #
        # The angular part is the link z axis: the tool is rigidly attached to
        # the last link, so it turns with it.
        f = self.dyn.frames(np.asarray(q, float))
        p_tool, _ = self._tool(f[-1])
        n = self.dyn.n
        J = np.zeros((6, n))
        for i, fi in enumerate(f):
            J[:3, i] = np.cross(fi['z'], p_tool - fi['p'])
            J[3:, i] = fi['z']
        return J


class URKinematics(AnalyticKinematics):
    # The hardware provider: chain FK/Jacobian plus analytic M(q)/g(q).
    #
    # `dyn` is an ArmDynamics built from the arm + payload link data.
    # `servo_gains` is the identified (Kp, Kv) of the real controller.
    # `tool_pos`/`tool_quat` are the gripper mount, wrist_3_link -> gripper base.

    def __init__(self, dyn, servo_gains, tool_pos=(0.0, 0.0, 0.0),
                 tool_quat=(1.0, 0.0, 0.0, 0.0)):
        super().__init__(ChainFrames(dyn, tool_pos, tool_quat), dyn, servo_gains)
