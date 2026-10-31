"""MuJoCo implementation of the ObjectWorld boundary.

This module is the SIMULATION side of the robot/world split. It answers the two
boundary questions -- "what can I see?" and "where is my tool?" -- from the
physics state, and it owns the object pool and the cube's demonstration motion.

The robot layer never imports this module. A simulation entry point constructs
it and hands it to the servo/task as an `ObjectWorld`; the hardware entry point
hands over a `RealObjectWorld` (src/world_real.py) instead.

SCENE LIFECYCLE
---------------
`spawn()`, `active()` and `pose()` are deliberately NOT on ObjectWorld: a
hardware world cannot teleport a cube into existence, so robot-side code must
not be able to call them. They exist here for the front-end and the harness.
"""

import numpy as np

from src.config import TOOL_MOUNT_POS
from src.world import ObjectWorld, TargetView


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
    def perceive(self) -> list:
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

    def gripper_pose(self):
        # The TCP, not the gripper/base body: the model's base body sits
        # TOOL_MOUNT_POS below tool0, and every height the servo works in is a
        # TCP height.
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        R = np.array(self.data.xmat[self.base_id], float).reshape(3, 3)
        p = (np.array(self.data.xpos[self.base_id], float)
             + R @ np.asarray(TOOL_MOUNT_POS, float))
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
