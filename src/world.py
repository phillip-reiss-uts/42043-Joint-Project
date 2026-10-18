# The boundary between the robot and everything else.
#
# Why this module exists
# ----------------------
# The robot side of this project (arm, servo, perception, observer, controller)
# must run on a real UR3e. The object side is pure demonstration: a cube that
# MuJoCo glides around the table so the tracking has something to chase. Those
# two concerns were previously tangled together -- Arm owned the cube's motion
# driver and stepped it inside its own trajectory loop, and the servo imported
# the object pool directly.
#
# That is fine in a simulator and impossible on hardware, where there is no
# `data.qvel` to write and no cube to teleport.
#
# So: robot code depends on this interface and nothing else. It may only
# OBSERVE the world through TargetView records, which are derived from the
# camera. It may never read ground-truth object state, and it may never write
# to an object at all.
#
# `SimObjectWorld` is the MuJoCo implementation. On hardware you would write a
# second implementation backed by RealSense captures, and the robot code would
# not change. Because the moving-target demo is simulation-only, the interface
# deliberately has no "move the object" method: nothing on the robot side ever
# needs to command the object, and giving it that capability is exactly how the
# two concerns got coupled in the first place.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


class SimObjectWorld(ObjectWorld):
    # MuJoCo implementation of the boundary.
    #
    # Owns the object pool and (the demonstration only) the cube's motion. The
    # mover is stepped by whoever drives the physics -- main.py's loop or the
    # test harness -- never by the arm. Robot-side code has no handle on it.

    def __init__(self, model, data, camera, cam_id, reg, mover=None):
        self.model = model
        self.data = data
        self.camera = camera
        self.cam_id = cam_id
        self.reg = reg

        # Demonstration-only: the cube glides so the servo has to chase it.
        # Present here so the sim world owns it, and absent from ObjectWorld so
        # no hardware implementation ever has to provide it.
        self.mover = mover

        self.base_id = self._resolve_base()

    def _resolve_base(self):
        import mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                'gripper/base')
        if bid < 0:
            raise ValueError("model has no 'gripper/base' body")
        return bid

    # OBSERVATION
    def perceive(self) -> List[TargetView]:
        # Camera-derived detections, converted to boundary records.
        from src.perception import perceive_all
        dets = perceive_all(self.model, self.data, self.camera, self.cam_id)
        return [TargetView(centre=np.array(d.centre, float),
                           yaw=float(d.yaw),
                           span=(None if getattr(d, 'span', None) is None
                                 else np.array(d.span, float)),
                           score=float(getattr(d, 'score', 0.0)),
                           n_points=int(d.n_points),
                           hull=(None if getattr(d, 'hull', None) is None
                                 else np.array(d.hull, float)))
                for d in dets]

    def gripper_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        p = np.array(self.data.xpos[self.base_id], float)
        R = np.array(self.data.xmat[self.base_id], float).reshape(3, 3)
        return p, R

    def camera_pose(self):
        # Camera pose in the world frame, for the servo trace. Not part of
        # ObjectWorld: the robot does not need to know where the camera is, only
        # where the tool is. This exists so debugging output can show the
        # frustum without the servo reaching into the simulator.
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        p = np.array(self.data.cam_xpos[self.cam_id], float)
        R = np.array(self.data.cam_xmat[self.cam_id], float).reshape(3, 3)
        return p, R

    # DEMONSTRATION DRIVER (simulation only)
    def step_mover(self):
        # Advance the cube's glide by one physics step. Called from the physics
        # loop, NOT from robot code.
        #
        # Note the mover receives no robot state: the cube's motion is
        # independent of where the arm is. This is the single call site for the
        # demonstration driver, which is what keeps it out of the robot.
        if self.mover is not None:
            self.mover.step(self.data)

    def set_mover(self, mover):
        self.mover = mover

    # OBJECT LIFECYCLE (simulation only)
    #
    # These exist for main.py and the harness to set the scene up. They are
    # deliberately NOT on ObjectWorld: a hardware world cannot teleport a cube
    # into existence, so robot-side code must not be able to call them.

    def spawn(self, rng, viewer=None):
        # Place the cube on the table. `world=self` so the settle loop keeps
        # advancing the demonstration motion while the cube falls.
        from src.pick_place import place_cube
        return place_cube(self.model, self.data, self.reg, rng, viewer=viewer,
                          settle=True, world=self)

    def active(self):
        # Which pooled objects are currently on the table. Object indices are a
        # simulation concept, so this stays off the boundary too.
        return self.reg.active()

    def pose(self, i):
        # Ground-truth object pose. FOR SCORING AND SCENE SETUP ONLY -- robot
        # code must never call this, and tests/test_layering.py enforces that.
        return np.array(self.reg.pose(self.data, i), float)
