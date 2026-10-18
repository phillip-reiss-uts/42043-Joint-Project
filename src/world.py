"""The boundary between the robot and everything else: the interface.

WHY THIS MODULE EXISTS
----------------------
The robot side of this project (arm, servo, perception, observer, controller)
must run on a real UR3e. The object side is pure demonstration: a cube that
MuJoCo glides around the table so the tracking has something to chase. Those
two concerns were previously tangled together -- Arm owned the cube's motion
driver and stepped it inside its own trajectory loop, and the servo imported
the object pool directly.

That is fine in a simulator and impossible on hardware, where there is no
`data.qvel` to write and no cube to teleport.

So: robot code depends on this interface and nothing else. It may only
OBSERVE the world through TargetView records, which are derived from the
camera. It may never read ground-truth object state, and it may never write
to an object at all.

IMPLEMENTATIONS
---------------
    src/world_sim.py    SimObjectWorld  -- MuJoCo, the demonstration
    src/world_real.py   RealObjectWorld -- RealSense depth, the hardware

Neither implementation is imported here. The entry point constructs one and
hands it to the robot layer, which is what keeps this module free of both
MuJoCo and a camera SDK.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TargetView:
    # One object as the robot is allowed to know it: a camera-derived estimate.
    #
    # This is the ONLY shape of object data that crosses the boundary. Note
    # what is absent -- there is no qpos, no true pose, no object handle. If a
    # robot-side routine needs something about the object, it has to be
    # expressible as something the camera could have measured.
    centre: np.ndarray                 # (3,) world position of the object centre
    yaw: float                         # degrees modulo 90
    span: Optional[np.ndarray] = None  # (2,) xy extent of the cluster
    score: float = 0.0                 # quality score; higher = cleaner, denser
    n_points: int = 0                  # points in the accepted cluster
    hull: Optional[np.ndarray] = None  # (M,3) top-face convex hull

    @property
    def ok(self):
        return self.centre is not None and self.n_points > 0

    @property
    def xy(self):
        return np.asarray(self.centre, float)[:2]


class ObjectWorld(ABC):
    # What the robot is allowed to ask about the world.
    #
    # Two methods, both read-only. Anything the robot side wants to know about
    # objects comes through perceive(), and anything it wants to know about
    # itself comes through gripper_pose(). Neither exposes a way to change the
    # world, which is what keeps the boundary honest.

    @abstractmethod
    def perceive(self) -> List[TargetView]:
        # Objects currently visible to the camera, best first.
        # An empty list means "nothing visible", not "nothing there".
        raise NotImplementedError

    @abstractmethod
    def gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        # (position, rotation matrix) of the gripper base, world frame.
        # The robot's own state, not the object's -- this is here so the servo
        # never has to reach into the simulator to find out where it is.
        raise NotImplementedError
