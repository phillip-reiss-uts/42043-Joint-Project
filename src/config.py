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

# object pool 
POOL_SIZE = 1
INITIAL_OBJECTS = 0       # spawned when the scene is reset
MIN_OBJECT_SEPARATION = 0.10   # centre-to-centre; keeps clusters from merging

# chute: sunk bin at the table's right edge, nearest the robot
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

# Gripper squeeze force limit (N), applied to the fingers actuator at build.
GRIP_FORCE_RANGE = (-5.0, 5.0)

# Viewer frame pacing: sync once per ~16 ms of simulated time (~60 Hz).
SYNC_DT = 0.016
SYNC_STEPS = 8      # SYNC_DT at the default 0.002 timestep; steps between syncs

# perception
SELF_FILTER = 0.20
SURFACE_BAND = 0.005
DEPTH_MAX = 5.0
MIN_OBJECT_POINTS = 1500
# LEFTOVER: unused -- superseded by the search inside clear_table.

# state observer (src/observer.py)
ENCODER_NOISE = 1.0e-4   # rad, std of joint-angle measurement noise (R)
TAU_NOISE = 0.30         # N.m, std of unmodelled joint torque (process noise)
OBSERVER_RELIN = 100     # steps between observer re-linearisations (0.2 s)
OBSERVER_SUBSTEPS = 4    # integration sub-steps/timestep (servo stiffness margin)

# camera calibration rig (src/calibration.py)
CALIB_MARKERS = 7           # markers per row and column (7x7 = 49)
CALIB_SPACING = 0.02        # marker pitch, metres
CALIB_MARKER_R = 0.003      # marker radius, metres
CALIB_BOARD_THICK = 0.004   # slab thickness, metres
CALIB_BOARD_MARGIN = 0.02   # slab overhang beyond the outer markers

# feedback control on the estimated state (src/controller.py)
USE_OBSERVER = True      # run the observer and close the loop on its estimate
FB_KQ = 3.0              # outer-loop position gain (dimensionless), PD fallback
FB_TD = 0.05             # outer-loop velocity gain (seconds), PD fallback
ENCODER_NOISY = True     # feed the observer a noisy encoder reading
USE_LQR = True           # design the outer-loop gains from the model (LQR)
                         # rather than using FB_KQ/FB_TD
LQR_Q_POS = 1.0          # state cost on position error (rad^2)
LQR_Q_VEL = 1.0e-4       # state cost on velocity error (rad/s)^2
LQR_R = 5.0e-2           # command cost (rad^2); sets the overall gain level
# Q_vel is the bandwidth knob; R alone barely moves the dominant pole.

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
HARD_SPAWNS = (
    (-0.219, 0.411),     # occluded by gripper: ~540 pts, 27 mm xy error
    (0.030, 0.400),      # occluded by gripper: ~1020 pts, 20 mm xy error
    (-0.200, 0.350),     # outside the home-pose footprint: 0 pts found
    (-0.320, 0.700),     # marginal footprint, just inside the far edge
)
HARD_SPAWN_MODE = False

# Spawns in the strip the home camera cannot see at all (footprint is y >= 0.354).
REALLY_HARD_MODE = False
HARD_BLIND_Y = (TABLE_CENTRE[1] - TABLE_HALF[1] + OBJECT_SIZE[1], 0.30)


# The object has a consitent 5.4 mm offset when placed, using an offset value does not affect this
PLACE_OFFSET_X = PLACE_OFFSET_Y = 0.0

# Revolve search: when the nominal view can't see the object, rotate the gripper
REVOLVE_DEG = tuple(range(15, 260, 15))
REVOLVE_SPEED = RAPID     # RAPID is safe here; the gripper is high, not near the object
SEARCH_SPEED = RAPID     # looks are not contact moves; the gripper is high

# PBVS commanded-twist caps, m/s. The servo law asks for kp*err directly, which
# is ~1.9 m/s at the start of an approach; unbounded that drives the gripper into
# the object and the table. Transit is the hover leg, descent the final approach.
SERVO_V_TRANSIT = 0.050
SERVO_V_DESCENT = 0.015
# Iteration budget per servo stage. At the caps above the hover leg alone needs
# ~3100 steps, so the default 1500 would time out and abort the pick.
SERVO_MAX_ITERS = 8000
# Iterations between servo re-observations. Each perceive_all blocks the loop
# for ~65 ms while a physics step costs 0.05 ms, so at the class default of 15
# the loop stalls almost continuously and the motion advances in 0.7 mm bursts.
# The estimate only moves ~0.2 mm per observation and the object is static, so
# observing less often costs almost nothing and lets the sim run.
SERVO_PERCEIVE_EVERY = 160

# Occlusion recovery. On the way down the jaws clip the object, the cluster's
# span shrinks, and the fitted centre drifts into the part still visible -- the
# servo then drives faithfully to a biased target and grasps off centre. Watch
# the span against the fullest view seen this stage; on a drop, back straight up
# until the view is whole again, then commit the grasp open loop from the
# fullest-view estimate instead of servoing into the occlusion.
SERVO_OCCLUSION_RECOVERY = True
SERVO_OCCLUDE_RATIO = 0.95      # span below this fraction of the best seen
SERVO_OCCLUDE_RECOVER = 0.97    # span must climb back above this to commit
SERVO_OCCLUDE_FAILS = 2         # or this many consecutive lost observations
SERVO_RETREAT_STEP = 0.02       # metres per backup step
SERVO_RETREAT_MAX = 15          # cap on backup steps
# Hover standoff for the closed-loop leg. Measured: at 100 mm above the grasp the
# jaws bias the fitted centre by ~0.55x any lateral gripper offset, in the same
# direction, so the estimate feeds back into the gripper and walks off centre.
# At 200 mm the bias is independent of offset (fully visible), so acquire and
# hold there and commit from that estimate.
SERVO_STANDOFF = 0.20

# Search offsets around the nominal gripper xy, tried in order.
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