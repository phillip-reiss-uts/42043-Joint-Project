# src/config.py
"""Single source of truth for scene and task constants."""

import numpy as np

# workcell 
TABLE_HEIGHT = 0.02
TABLE_THICK = 0.02
TABLE_CENTRE = [-0.134, 0.492]
TABLE_HALF = [0.35, 0.35]

OBJECT_SIZE = [0.03, 0.03, 0.03]
OBJECT_HEIGHT = 2 * OBJECT_SIZE[2]
OBJECT_POS = [TABLE_CENTRE[0], TABLE_CENTRE[1], TABLE_HEIGHT + OBJECT_SIZE[2]]

# camera mount 
CAM_QUAT = [0.0, 0.707, 0.707, 0.0]
CAM_POS = [0.05, 0.0, -0.03]
MESH_QUAT = [0.707, 0.0, 0.0, 0.707]
MESH_POS = CAM_POS

# arm / gripper
ARM_DOF = 6
GRIP_OPEN = 0.0
GRIP_CLOSED = 255.0
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])

# motion rates (CNC style)
RAPID = 0.20
FEED = 0.015
RAPID_MIN_STEPS = 200
FEED_MIN_STEPS = 400

# perception
SELF_FILTER = 0.20
SURFACE_BAND = 0.005
DEPTH_MAX = 5.0
MIN_OBJECT_POINTS = 1500
PERCEPTION_TRIES = 5
VIEW_Y_LEAD = 0.05
VIEW_OFFSETS = ((0.0, 0.0), (0.0, VIEW_Y_LEAD), (0.0, -VIEW_Y_LEAD))

# grasp / place 
GRASP_HEIGHT_FRAC = 0.2
TIP_OFFSET = 0.1094 - (GRASP_HEIGHT_FRAC - 0.5) * OBJECT_HEIGHT
APPROACH_Z = 0.10
FINAL_Z = 0.03
LIFT_Z = 0.15
PLACE_DX = 0.0
PLACE_DY = 0.0
PLACE_CLEARANCE = 0.002
STEPS_CLOSE = 3000
STEPS_RELEASE = 2000

# spawn parameters
RANDOMIZE_SPAWN = True
RANDOMIZE_YAW = True     # spin the object about z
SPAWN_CLEARANCE = 0.032  # spawn height above the surface; it settles onto it
SPAWN_MARGIN = 0.06      # keep this clear of the table edge
SPAWN_Y_MIN = 0.45       # measured: below this the object leaves the camera footprint

SPAWN_X_RANGE = (TABLE_CENTRE[0] - TABLE_HALF[0] + SPAWN_MARGIN + OBJECT_SIZE[0],
                 TABLE_CENTRE[0] + TABLE_HALF[0] - SPAWN_MARGIN - OBJECT_SIZE[0])
SPAWN_Y_RANGE = (max(TABLE_CENTRE[1] - TABLE_HALF[1] + SPAWN_MARGIN + OBJECT_SIZE[1],
                     SPAWN_Y_MIN),
                 TABLE_CENTRE[1] + TABLE_HALF[1] - SPAWN_MARGIN - OBJECT_SIZE[1])

# Deliberately awkward spawns for exercising the perception-recovery path.
# Measured against this scene. (x, y) only; height comes from SPAWN_CLEARANCE.
HARD_SPAWNS = (
    (-0.219, 0.411),     # occluded by gripper: ~540 pts, 27 mm xy error
    (0.030, 0.400),      # occluded by gripper: ~1020 pts, 20 mm xy error
    (-0.200, 0.350),     # outside the home-pose footprint: 0 pts found
    (-0.320, 0.700),     # marginal footprint, just inside the far edge
)
HARD_SPAWN_MODE = False

# Spawns in the strip the home camera cannot see at all (footprint is y >= 0.354).
# KNOWN FAILING: perception returns a table-sized cluster that the bbox guard
# rejects, so every attempt exhausts PERCEPTION_TRIES. Kept as documentation of
# the limit rather than a working mode.
REALLY_HARD_MODE = False
HARD_BLIND_Y = (TABLE_CENTRE[1] - TABLE_HALF[1] + OBJECT_SIZE[1], 0.30)

# debug flags
VERIFY_IK = True         # run the IK self-test at startup
VERBOSE = True           # per-stage [EXEC]/[PERCEPTION] logging