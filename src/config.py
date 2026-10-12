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

# object pool -------------------------------------------------------------
# Pre-compiled objects; bodies are created once at model build and activated
# by moving them from the parking pose onto the table. MuJoCo cannot add
# bodies after compile, so this pool is the ceiling on simultaneously
# spawnable objects.
POOL_SIZE = 8
INITIAL_OBJECTS = 4       # spawned when the scene is reset
MIN_OBJECT_SEPARATION = 0.10   # centre-to-centre; keeps clusters from merging

# chute: attached to the table's right edge, nearest the robot, sunk below
# the tabletop. Dropped objects land in it and despawn until R.
CHUTE_X = (0.216, 0.566)        # x extent; left edge touches the table edge
CHUTE_Y = (0.142, 0.492)        # y extent; 1/4 of the table's area
CHUTE_FLOOR_Z = 0.02 - 0.35     # floor 35 cm below the tabletop
CHUTE_WALL_THICK = 0.03         # wall thickness, half-extent in x/y
CHUTE_WALL_HEIGHT = 0.35        # walls rise to tabletop height

# pool parking pose: far below the scene; objects are frozen here until spawned
PARK_POS = [5.0, 5.0, -0.6]

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
FEED = 0.05
RAPID_MIN_STEPS = 200
FEED_MIN_STEPS = 400

# Gripper squeeze force limit, newtons, applied to the fingers actuator
# menagerie default is [-5, 5], so this changes nothing until edited.
GRIP_FORCE_RANGE = (-5.0, 5.0)

# Viewer frame pacing. Sync once per ~16 ms of simulated time (~60 Hz), rather
# than every N steps or a fixed number of frames per move: a fixed step interval
# stutters on a short fast hop (a 60 mm move at RAPID is ~30 steps, so syncing
# every 25 renders one frame) while a fixed frame count burns hundreds of
# renders on a long slow one. Each viewer.sync() blocks on a render, so this is
# the dominant cost of a search: measured, it was worth seconds per attempt.
SYNC_DT = 0.016
SYNC_STEPS = 8      # SYNC_DT at the default 0.002 timestep; steps between syncs

# perception
SELF_FILTER = 0.20
SURFACE_BAND = 0.005
DEPTH_MAX = 5.0
MIN_OBJECT_POINTS = 1500
# LEFTOVER: unused -- the singleton peek/revolve path these fed was replaced by
# the search inside clear_table.
# PERCEPTION_TRIES = 5
# VIEW_Y_LEAD = 0.05
# VIEW_OFFSETS = ((0.0, 0.0), (0.0, VIEW_Y_LEAD), (0.0, -VIEW_Y_LEAD))

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
DROP_HOLD_STEPS = 500      # hold the fingers open after a drop before the next pick

# spawn parameters
RANDOMIZE_SPAWN = True
RANDOMIZE_YAW = True     # spin the object about z
SPAWN_CLEARANCE = 0.05  # spawn height above the surface
SPAWN_MARGIN = 0.06      # keep this clear of the table edge
SPAWN_Y_MIN = 0.45       # measured: below this the object leaves the camera footprint
SPAWN_ARM_CLEARANCE = 0.12   # instant drops keep this clear of gripper/base (xy)

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
# Currently fails, every attempt exhausts PERCEPTION_TRIES. 
REALLY_HARD_MODE = False
HARD_BLIND_Y = (TABLE_CENTRE[1] - TABLE_HALF[1] + OBJECT_SIZE[1], 0.30)


# The object has a consitent 5.4 mm offset when placed, using an offset value does not affect this
PLACE_OFFSET_X = PLACE_OFFSET_Y = 0.0

# Revolve search: when the nominal view can't see the object, rotate the gripper
# at a fixed standoff. The camera sits behind gripper/base while the jaws rotate,
# so sweeping yaw changes which gap the camera looks through without changing
# standoff. Measured: clears 19/20 spawns, first clearing angle 0-60 deg,
# mean 8 deg -- usually the nominal view alone works.
REVOLVE_DEG = tuple(range(15, 260, 15))
REVOLVE_SPEED = RAPID     # RAPID is safe here; the gripper is high, not near the object
SEARCH_SPEED = RAPID     # looks are not contact moves; the gripper is high

# Search offsets around the nominal gripper xy, tried in order.
# Measured: for objects at the blind edge (y 0.35-0.43) the offsets that expose
# them are +y, -x and the diagonals -- NOT -y. An earlier list of (0,-0.08),
# (0,-0.16), (0,-0.24) searched the wrong half entirely.
SEARCH_OFFSETS = (
    (0.0, 0.0),        # nominal, retried after any arm settling
    (0.0, +0.06),      # back -- puts the camera over the object
    (-0.10, 0.0),      # left
    (+0.10, 0.0),      # right
    (0.0, +0.12),
    (-0.10, +0.06),
    (+0.10, +0.06),
    (-0.10, +0.12),
    (+0.10, +0.12),
    (0.0, -0.06),      # forward last: rarely helps, kept for completeness
)

DROP_RADIUS = 0.07        # scatter drops this far around the chute centre

# debug flags
VERIFY_IK = False         # run the IK self-test at startup
VERBOSE = False           # per-stage [EXEC]/[PERCEPTION] logging