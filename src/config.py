# src/config.py
"""Single source of truth for scene and task constants."""

import numpy as np

# workcell
#
# GEOMETRY IS TUNED FOR THE UR3e (~500 mm reach), not the UR5e this project
# started on (~850 mm).
#
# The table is a 400 mm square. The UR3e cannot reach a table the size of the
# original one (700 x 700 mm, centre [-0.134, 0.492]) -- that put most of the
# spawn region 0.4-0.5 m from the base, out of reach, so the servo never saw the
# object. 400 mm is the largest SQUARE that keeps the whole spawn region
# reachable; the reachable-down region measured by sweeping IK against
# models/ur3e/ur3e.xml is a wedge, and the table's outer margin falls outside
# it at the near corner (x < ~0.10) where the arm cannot fold back.
#
# Checked: with SPAWN_MARGIN and the object half-extent applied, the spawn
# region x 0.165..0.385, y -0.075..0.145 is 25/25 reachable bottom-up in a
# straight-down pose. Only the unused margin strips are unreachable.
#
# The centre is unchanged from the UR3e re-tune, so the home keyframe in
# models/ur3e/ur3e.xml still hovers over the middle of the table.
#
# If you switch back to the UR5e, restore:
#     TABLE_CENTRE = [-0.134, 0.492]   TABLE_HALF = [0.35, 0.35]
#     CHUTE_X = (0.216, 0.566)         CHUTE_Y = (0.142, 0.492)
TABLE_HEIGHT = 0.02
TABLE_THICK = 0.02
TABLE_CENTRE = [0.275, 0.035]
TABLE_HALF = [0.20, 0.20]

OBJECT_SIZE = [0.03, 0.03, 0.03]
OBJECT_HEIGHT = 2 * OBJECT_SIZE[2]
OBJECT_POS = [TABLE_CENTRE[0], TABLE_CENTRE[1], TABLE_HEIGHT + OBJECT_SIZE[2]]

# object pool 
POOL_SIZE = 1
INITIAL_OBJECTS = 0       # spawned when the scene is reset
MIN_OBJECT_SEPARATION = 0.10   # centre-to-centre; keeps clusters from merging

# NO CHUTE. There is one flat table and nothing else: a picked object is set
# back down on the table, and the table is never cleared into a bin. The
# original design had a sunk chute cut into the table's far edge, but it was
# a simulation-only convenience -- a real cell has no such bin, and dropping
# into a hole is a different motion from placing.
#
# The removal took CHUTE_X/CHUTE_Y/CHUTE_FLOOR_Z/CHUTE_WALL_THICK/
# CHUTE_WALL_HEIGHT with it, plus the chute geoms in the MJCF and the
# despawn-into-chute bookkeeping. DROP_RADIUS survives; it now scatters the
# release point across a patch of tabletop.
#
# If a bin is ever wanted again, put it OUTSIDE the table footprint and give
# the drop its own reachability check -- the old one sat inside the table and
# half its scatter radius fell off the edge.

# where a released object is put down, relative to the table
PLACE_EDGE_MARGIN = 0.10   # keep this clear of the far edge so it stays in view

# pool parking pose: far below the scene; objects are frozen here until spawned
PARK_POS = [5.0, 5.0, -0.6]

# camera mount 
CAM_QUAT = [0.0, 0.707, 0.707, 0.0]
CAM_POS = [0.05, 0.0, -0.03]
MESH_QUAT = [0.707, 0.0, 0.0, 0.707]
MESH_POS = CAM_POS

# arm / gripper / camera identity
#
# HARDWARE TARGET: real UR3e + RealSense D435i + Robotiq 2F-85.
#
# The UR3e is now simulated directly. mujoco_menagerie still ships no UR3e, so
# the official UR3e MJCF is vendored at models/ur3e/ur3e.xml (sourced from the
# UR3_RG2_rosbridge teaching platform, which carries the manufacturer's link
# inertials). model_loader resolves a name under models/ first and only then
# falls back to menagerie, so 'ur3e' selects the local file.
#
# The UR5e the project was originally developed against is still available:
#     ROBOT_ARM = 'universal_robots_ur5e'
# Expect the tuned workspace constants (spawn ranges, table/chute geometry,
# standoff) to need re-fitting if you switch arms: the UR3e has ~500 mm reach
# against the UR5e's ~850 mm.
ROBOT_ARM = 'ur3e'                       # models/ur3e/ur3e.xml
ROBOT_GRIPPER = 'robotiq_2f85'           # Robotiq 2F-85 (the "2D-85" in the
                                         # brief is a typo; 2F-85 is the model)
ROBOT_CAMERA = 'realsense_d435i'         # assumed wrist-mounted D435i

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
# LEFTOVER: ENCODER_NOISY is referenced nowhere; the observer always uses the
# clean encoder reading.
# ENCODER_NOISY = True
USE_LQR = True           # design the outer-loop gains from the model (LQR)
                         # rather than using FB_KQ/FB_TD
LQR_Q_POS = 1.0          # state cost on position error (rad^2)
LQR_Q_VEL = 1.0e-4       # state cost on velocity error (rad/s)^2
LQR_R = 5.0e-2           # command cost (rad^2); sets the overall gain level
# Q_vel is the bandwidth knob; R alone barely moves the dominant pole.

# grasp / place
#
# TIP_OFFSET places the gripper/base when grasping: servo.desired_pose sets
#     p_des.z = object_centre_z + TIP_OFFSET + z_offset
# so this is where the tool is commanded relative to the object centre.
#
# NOTE ON THE FORMULA. It is NOT simply "distance from gripper/base to the pad".
# Measuring that (0.1401 m on both builds) and using base_to_tip - H/2 gives
# 0.1101, which was tried and made grasps WORSE: the servo then drives closer
# before the tracking gate fires, leaving under 50 mm for the open-loop commit,
# which truncates and closes the jaws short of the cube. Measured commit offsets
# from the target: 0.042 / 0.011 / 0.054 m at 0.1101, versus -0.002 / -0.005 /
# 0.034 m at the original 0.1274. The original value is retained.
#
# The 0.1094 term is an early empirical figure. It is left as-is rather than
# "corrected" because the surrounding behaviour (track-then-commit) is tuned
# against it, and changing it silently changes where the servo stops.
GRASP_HEIGHT_FRAC = 0.2
TIP_OFFSET = 0.1094 - (GRASP_HEIGHT_FRAC - 0.5) * OBJECT_HEIGHT
BASE_TO_TIP = 0.1401          # measured, gripper/base -> lowest pad (reference)
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
# UR5e-era footprint floor. The UR5e's home camera sat ~0.49 m above the table
# and could not see below y = 0.45. The re-tuned UR3e workcell is centred at
# y = 0.035, so this floor would invert the spawn range entirely. Disabled
# (None) for the UR3e; the reachable rectangle already bounds the region and
# tests/test_hardware.py checks the camera can see the samples it produces.
# Restore 0.45 if switching back to the UR5e table.
SPAWN_Y_MIN = None
SPAWN_ARM_CLEARANCE = 0.12   # instant drops keep this clear of gripper/base (xy)

# The spawn region is the intersection of what the arm can REACH and what the
# wrist camera can SEE from the home pose -- not just the table's inner rect.
# With the table widened to 400 mm those stopped coinciding: the camera sits on
# the flange and looks at the table centre, so an object near the table's near
# edge (x < ~0.24) is invisible from home even though the arm could reach it.
# Measured on the 400 mm table at the home pose:
#     visible at y = 0.035: x 0.24..0.46
#     visible at x = 0.30 : y -0.15..0.21
# Intersected with the reachable set this gives x 0.22..0.46, y -0.15..0.21.
# SPAWN_X_MIN clamps the low x end to that; the upper bounds follow the table
# edge with SPAWN_MARGIN and the object half-extent applied.
SPAWN_X_MIN = 0.24       # measured: below this the object leaves the home view

_x_lo = TABLE_CENTRE[0] - TABLE_HALF[0] + SPAWN_MARGIN + OBJECT_SIZE[0]
_y_lo = TABLE_CENTRE[1] - TABLE_HALF[1] + SPAWN_MARGIN + OBJECT_SIZE[1]
SPAWN_X_RANGE = (max(_x_lo, SPAWN_X_MIN),
                 TABLE_CENTRE[0] + TABLE_HALF[0] - SPAWN_MARGIN - OBJECT_SIZE[0])
SPAWN_Y_RANGE = ((_y_lo if SPAWN_Y_MIN is None else max(_y_lo, SPAWN_Y_MIN)),
                 TABLE_CENTRE[1] + TABLE_HALF[1] - SPAWN_MARGIN - OBJECT_SIZE[1])

# Deliberately awkward spawns for exercising the perception-recovery path.
#
# Re-expressed for the UR3e workcell. The original UR5e coordinates
# ((-0.219,0.411), (0.030,0.400), (-0.200,0.350), (-0.320,0.700)) are outside
# the UR3e's reach entirely, so they exercised "cannot reach" rather than
# "cannot see". These sit inside the re-tuned reachable rectangle and are hard
# for the stated reason: near the inside reach limit, at the edges of the
# camera footprint, and in the corner where the wrist approaches a singularity.
# The exact point counts differ from the UR5e measurements; re-measure with
# HARD_SPAWN_MODE = True if the numbers matter for a report.
HARD_SPAWNS = (
    (0.180, -0.140),     # near the inside reach limit, gripper partly occludes
    (0.320, 0.240),      # far y edge, marginal camera footprint
    (0.230, 0.000),      # close to the base, wrist near a singularity
    (0.390, -0.150),     # outer x limit, close to the chute
)
HARD_SPAWN_MODE = False

# Spawns in the strip the home camera cannot see at all.
# The home camera sits above the gripper looking down, so the blind strip is the
# region nearest the base, not the far edge. Bounds follow the re-tuned table.
REALLY_HARD_MODE = False
HARD_BLIND_Y = (TABLE_CENTRE[1] - TABLE_HALF[1] + OBJECT_SIZE[1],
                TABLE_CENTRE[1] - TABLE_HALF[1] + 0.10)


# The object has a consitent 5.4 mm offset when placed, using an offset value does not affect this
PLACE_OFFSET_X = PLACE_OFFSET_Y = 0.0

# Revolve search: when the nominal view can't see the object, rotate the gripper
REVOLVE_DEG = tuple(range(15, 260, 15))
REVOLVE_SPEED = RAPID     # RAPID is safe here; the gripper is high, not near the object
SEARCH_SPEED = RAPID     # looks are not contact moves; the gripper is high

# PBVS commanded-twist caps, m/s. The servo law asks for kp*err directly, which
# is ~1.9 m/s at the start of an approach; unbounded that drives the gripper into
# the object and the table. Transit is the hover leg, descent the final approach.
# Transit must exceed the fastest object band, or the arm can never close the
# gap on a moving target -- at 0.05 m/s it merely matched the medium band and
# tracked nothing. It runs in free space, well above the object, so it is safe
# to be brisk; only the descent needs to be gentle.
SERVO_V_TRANSIT = 0.150
SERVO_V_DESCENT = 0.015
# Iteration budget per servo stage. At the caps above the hover leg alone needs
# ~3100 steps, so the default 1500 would time out and abort the pick.
SERVO_MAX_ITERS = 8000
# Iterations between servo re-observations. Each perceive_all blocks the loop
# for ~65 ms while a physics step costs 0.05 ms, so at the class default of 15
# the loop stalls almost continuously and the motion advances in 0.7 mm bursts.
# The estimate only moves ~0.2 mm per observation and the object is static, so
# observing less often costs almost nothing and lets the sim run.
SERVO_PERCEIVE_EVERY = 100

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
# Cartesian speed for the backup and the open-loop commit, m/s. arm.move paces
# on joint distance with a rad/s rate, so SERVO_RETREAT_V is converted through
# the joint distance each move needs; at the RAPID rate the backup ran ~80 mm/s
# against the servo's 15 mm/s descent.
SERVO_RETREAT_V = 0.03
# Hover standoff for the closed-loop leg. Measured: at 100 mm above the grasp the
# jaws bias the fitted centre by ~0.55x any lateral gripper offset, in the same
# direction, so the estimate feeds back into the gripper and walks off centre.
# At 200 mm the bias is independent of offset (fully visible), so acquire and
# hold there and commit from that estimate.
SERVO_STANDOFF = 0.20

# ---- moving target (robustness demonstration) ----
# The object glides on the table while the arm servos, so the servo has to
# chase. Bands are toggled live with the '.' key; index 0 is off. All bands sit
# below SERVO_V_TRANSIT so the arm can actually close the gap.
OBJ_MOVE_BANDS = (
    ('off',    0.000),
    ('slow',   0.015),
    ('medium', 0.030),
    ('fast',   0.045),
)
# LEFTOVER: the live band lives in target_motion._state, which is what the .
# key cycles. This constant was never read.
# OBJ_MOVE_BAND = 0
OBJ_MOVE_TURN_S = 1.2        # seconds between random re-aims
# REMOVED: OBJ_MOVE_STOP_DIST (proximity halt). It made the cube's motion law
# depend on the robot's position, which broke the robot/world separation, and
# it was already disabled (0.0). The robot now waits for the object to hold
# still instead of forcing it to stop, so the parameter has no meaning.

# ---- vision Kalman filter ----
# Smooths the intermittent vision stream, bridges the stretches where the
# object is invisible, and supplies a velocity for the lead term.
SERVO_USE_KF = True
KF_SIGMA_MEAS = 1.5e-3       # measurement noise, m (320x240 is ~1.5 mm)
KF_SIGMA_ACC = 0.60          # manoeuvre noise, m/s^2 (band changes + re-aims)
KF_GATE = 3.0                # innovation gate, sigmas; 0 disables
# Aim ahead of the estimate by this long to cancel the servo's own lag. Only
# has an effect once the filter sees a velocity, so a still object is unaffected.
SERVO_LEAD_S = 0.15

# ---- integral + damping on the pose error ----
#
# BOTH DEFAULT TO OFF, and should stay off unless there is a specific reason.
#
# They were added while chasing a sag-and-jerk that turned out to be a bug in
# servo(): the commanded joint vector was re-seeded from data.qpos on every
# call, so chunking the loop ratcheted the command downhill (measured: 600
# calls of 1 iteration drifted 102 mm; one call of 600 iterations drifted 1.8
# mm). With that fixed, both terms measurably HURT:
#
#   ki    damp_gain   reversals   qvel_sd   z_err
#   0.0   0.0         9           0.0023    0.05 mm   <- best, used
#   0.5   0.0         4           0.0034    0.16 mm
#   0.0   1.0         9           0.0038    0.19 mm
#   0.5   1.0         5           0.0043    0.17 mm
#
# (Reversals are noisy at this scale -- qvel_sd ~0.003 rad/s is near the noise
# floor -- but the error column is unambiguous: the integral raises it 3x,
# because it integrates against a bias that is no longer there.)
#
# They are kept, defaulted off, because on hardware the gravity feedforward is
# approximate rather than exact, and a standing bias is exactly what the I term
# is for. Re-enable and re-tune then; do not assume these numbers transfer.
SERVO_KI = 0.0               # integral gain on the Cartesian pose error
SERVO_KI_NEAR_M = 0.05
SERVO_KI_NEAR_RAD = 0.30
SERVO_KI_MAX_M = 0.02
SERVO_KI_MAX_RAD = 0.20
SERVO_DAMP_GAIN = 0.0        # joint-space D; 0 disables
SERVO_DAMP_NEAR_M = 0.05
SERVO_DAMP_NEAR_RAD = 0.15

# ---- tracking a moving target ----
# The robot servos for as long as the object keeps moving, re-acquires it with
# the search routine when it leaves the view, and commits to the open-loop
# pickup only once it has watched the object hold still. Committing on stillness
# rather than on proximity is what lets the chase run as long as it needs to,
# and it removes the servoed descent into the jaws' own shadow entirely -- the
# grasp is driven open loop from the hover, where the view is still whole.
SERVO_TRACK_MOVING = True
SERVO_STILL_TIME = 3.0        # s of observed stillness before committing
SERVO_STILL_TOL = 0.008       # m; filtered position may drift this much and
                              # still count as stationary. 8 mm over 3 s is an
                              # effective speed gate of ~2.7 mm/s, well clear of
                              # the filter's sub-millimetre position noise.
SERVO_STILL_SPEED = 0.006     # m/s; used only to label the target in the overlay
SERVO_TRACK_TOL = 0.020       # m; must also be this close before committing
SERVO_TRACK_TOL_ROT = 0.15    # rad
SERVO_TRACK_ITERS = 60000     # iteration cap on the chase (~2 min of sim)
SERVO_LOST_LIMIT = 5          # consecutive failed looks before searching
SERVO_SEARCH_LIMIT = 2        # give up after this many fruitless searches
# Watch the target through the open-loop descent, using the camera only. If it
# starts moving again, abandon the descent and go back to tracking: driving to
# where the object used to be is not a grasp.
SERVO_WATCH_DESCENT = True
# Filtered position drift from the commit reference that counts as moving.
# 15 mm is ~10x the sub-2 mm estimate noise at the hover and far above the ~1 mm
# depth bias on a straight descent, so the robot's own approach does not trip
# it. At the slowest band that is one second of travel, so the motion is
# unmistakable before it fires.
SERVO_MOVE_TOL = 0.015
SERVO_COMMIT_TRIES = 5        # abandon-and-retrack cycles before giving up

# ---- grasp verification ----
# Read the jaws back after closing, instead of assuming a completed approach
# means a completed grasp. These are the gripper's own registers -- finger
# position and actuator force, the same pair a real 2F-85 reports over Modbus --
# so nothing here depends on knowing where the object is.
GRASP_DRIVER_JOINTS = ('gripper/right_driver_joint', 'gripper/left_driver_joint')
# Measured on this model: closing on nothing drives the driver joints to 0.77
# (range 0..0.8) with the actuator force at -0.13 as the linkage bottoms out.
# Closing on the 30 mm cube stalls them at 0.25 with the force saturated at
# +5.0 and the finger velocity at ~0.
GRASP_CLOSED_Q = 0.60         # above this the jaws plainly closed on nothing
GRASP_FORCE_MIN = 1.0         # force needed to call it gripped, not just touched
GRASP_VEL_TOL = 0.02          # rad/s; below this the fingers count as stalled
GRASP_LIFT_RECHECK = 0.04     # m; lift clear and re-read after a failed check
# Yaw blend gain, separate from the position blend. The top face is square, so
# the calipers can label the same orientation as yaw or yaw+90. With the raw
# reading the label flipped every look, commanding a ~90 deg wrist swing and
# leaving err_rot stuck at 0.59 rad -- above every commit gate.
SERVO_YAW_SMOOTH = 0.35

# Hold the simulation to wall-clock real time in the viewer. Without this the
# step loops free-run at whatever the renderer allows -- ~0.2x real time while
# waiting, ~2-3x during a servo -- so the same motion looks glacial in one phase
# and lurching in the other. Pacing also re-anchors after a perception stall
# rather than sprinting to catch up.
VIEWER_REAL_TIME = True

# Live point-cloud panel in the viewer, drawn beside the camera image: the
# points the depth pipeline kept, projected back into the camera plane and
# coloured by range. It needs its own depth render, so it refreshes on its own
# slower cadence -- the viewer syncs every 8 steps during a servo, and matching
# that would add a depth pass 3x more often than perception itself runs.
OVERLAY_CLOUD = True
OVERLAY_CLOUD_EVERY = 30      # viewer syncs between cloud refreshes


# Upper bound on clear_table passes. A pick can legitimately fail (the object
# was never visible, the servo gave up); without this the task loop re-homes and
# retries forever.
MAX_PASSES = 12

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