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
# MEASURED ON THE REAL CELL (2026-10-23), not the old simulated workcell.
# The table was moved relative to the robot: it now sits BEHIND the base in -y,
# not in front, and it is rectangular, axis-aligned with the robot's x/y (the
# long edge measured along x to 0.000 deg).
#
#   far edge walked with the tool:  A (-0.2200, -0.4200) -> B (+0.2200, -0.4200)
#   long edge 440 mm, depth 300 mm, so the centre is 150 mm in from that edge.
#
# The robot is BOLTED TO THE TABLE, so the plane objects rest on is the base
# plane: TABLE_HEIGHT = 0. Model_loader puts the arm base at the world origin, so
# the tabletop is z = 0 and nothing needs raising.
#
# The 0.2315 that used to live here is NOT that height. It is the distance from
# the FLANGE down to the tabletop when the gripper's tip is placed against the
# table (measured as 0.2423 to the flange, minus TOOL_MOUNT's 10.8 mm). That is a
# TOOL dimension -- how long the gripper is when pointing down -- and it belongs
# with the other tool geometry below, not with the work surface.
#
# RESOLVED 2026-10-25: the tool is 231.5 mm from the flange to the LOWEST PAD,
# which is the 0.2315 above. BASE_TO_TIP (0.1401) was measured to the pad's
# gripping edge rather than the fingertip, so it is a shorter number describing a
# different point. Both are right; they were never measuring the same thing. The
# simulated RG2 is now built to 231.5 mm.
#
# NOT YET ACTED ON: TIP_OFFSET is tuned against grasp heights that were derived
# while the tool was modelled at 149 mm. It has NOT been retuned for a 231.5 mm
# tool, so expect the commit to sit high until it is -- and that is a measurement
# at the cell, not a guess here.
TABLE_HEIGHT = 0.0
TABLE_THICK = 0.02
TABLE_CENTRE = [0.0, -0.270]
TABLE_HALF = [0.22, 0.15]

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
#
# Origin is gripper/base. Local +z runs DOWN the approach, toward the finger tips,
# so CAM_POS z is measured downward from the mount face: -0.01 is 10 mm UP towards
# the wrist, and x = +0.05 puts the camera 50 mm off the tool axis.
#
# The camera cannot mount to the wrist itself -- cables and connectors occupy that
# space -- but this offset is retained. A centred placement was tried at
# (0, 0, 0), which put the lens level with the top face of the gripper housing and
# 50 mm inboard of the working offset, and was rejected.
#
# MOVED toward the finger tips in two steps: z -0.03 -> -0.01 -> +0.09, 120 mm in
# total. x, y and the rotation are unchanged. At +0.09 the lens is 90 mm down the
# approach, level with the gripper housing -- the housing meshes occupy that space,
# so expect them in view.
CAM_QUAT = [0.0, 0.707, 0.707, 0.0]
CAM_POS = [0.05, 0.0, 0.09]
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
# The real gripper is an OnRobot RG2, so the simulation models one too:
# models/onrobot_rg2/onrobot_rg2.xml (a parallel-jaw proxy with the RG2's 110 mm
# stroke, 0.9 kg mass and the same 0-255 command space). It used to be a Robotiq
# 2F-85, which has an 85 mm stroke and different fingers -- every simulated
# conclusion about jaw opening was about hardware that is not on the bench.
ROBOT_GRIPPER = 'onrobot_rg2'            # models/onrobot_rg2/onrobot_rg2.xml
ROBOT_CAMERA = 'realsense_d435i'         # assumed wrist-mounted D435i

ARM_DOF = 6
GRIP_OPEN = 0.0
GRIP_CLOSED = 255.0
# RG2 linkage. The official URDF's driven quantity is finger_width, an OPENING
# width of 0..0.110 m (0 = SHUT), and the knuckle angle follows it through
#   finger_joint = 0.785398 - 10.38471 * finger_width
# so the knuckle sweeps -0.35692 rad (open) .. +0.785398 rad (shut).
#
# MEASURED on the composed model with the real meshes (mj_geomDistance between the
# two finger pads): 97.3 mm at the open end, 0 mm at the shut end. The URDF's own
# "110 mm" is the width parameter, not the clear pad gap.
#
# Note the direction: finger_width 0 is SHUT, the opposite of the hardware command
# space (0 = open). MujocoBackend does that inversion so GRASP_CLOSED_Q keeps
# meaning "fraction of the way closed" on both paths.
GRIP_FINGER_OPEN_ANGLE = -0.35692
GRIP_FINGER_SHUT_ANGLE = 0.785398
GRIP_FINGER_MIMIC_GAIN = 10.38471     # rad of knuckle per metre of width
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])

# motion rates (CNC style)
RAPID = 0.20
FEED = 0.05
RAPID_MIN_STEPS = 200
FEED_MIN_STEPS = 400

# Gripper squeeze force limit (N), applied to the fingers actuator at build.
#
# The RG2 is rated to 40 N, and the official linkage needs that headroom: at the
# old +/-5 N the actuator could not hold the mimic chain open, so the jaws sat at
# an arbitrary half-open pose instead of following the command. The hardware path
# uses RG2_GRIP_FORCE_N for the real gripper, so this only bounds the simulation.
GRIP_FORCE_RANGE = (-40.0, 40.0)

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
#
# SPAWN_X_MIN used to clamp the low-x end to 0.24, measured on the OLD 400 mm
# workcell whose table was centred at x = 0.275. With the measured CELL values
# (table centred at x = 0, half-width 0.22) that clamp sits outside the table and
# INVERTS the range: max(-0.13, 0.24) == 0.24 > 0.13, so rng.uniform raised
# "high - low < 0" and no trial could start. Disabled until the visible-from-home
# region is re-measured on this cell.
SPAWN_X_MIN = None       # re-measure on THIS cell before setting it again

_x_lo = TABLE_CENTRE[0] - TABLE_HALF[0] + SPAWN_MARGIN + OBJECT_SIZE[0]
_x_hi = TABLE_CENTRE[0] + TABLE_HALF[0] - SPAWN_MARGIN - OBJECT_SIZE[0]
_y_lo = TABLE_CENTRE[1] - TABLE_HALF[1] + SPAWN_MARGIN + OBJECT_SIZE[1]

if SPAWN_X_MIN is not None:
    _x_lo = max(_x_lo, SPAWN_X_MIN)
if _x_hi <= _x_lo:
    # Fail here with a readable message rather than inside numpy.random.
    raise ValueError(
        'spawn x range is empty: lo %.3f >= hi %.3f. SPAWN_X_MIN and the table '
        'constants disagree about which workcell this is -- fix the config, do '
        'not clamp around it.' % (_x_lo, _x_hi))

SPAWN_X_RANGE = (_x_lo, _x_hi)
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
#
# RAISED to 350 mm to put a palm-mounted camera inside the D435i's 0.3-3 m ideal
# range, then put BACK to 200 mm after measuring both in the sim, 6 trials each:
#
#     standoff   approach complete   picked   wrist estimate error
#     0.20 m          6/6              2/6      5-10 mm
#     0.35 m          2/6              1/6      1.5-2 mm
#
# Raising it does help the estimate, as expected -- the target is further from the
# jaws and less occluded -- but it also lengthens the OPEN-LOOP commit that
# follows, and that is where the grasp error comes from: commit gaps of 14-29 mm
# leave the jaws closing on air. Net, 350 mm costs four approaches out of six.
#
# So the earlier assumption that raising it was free was wrong. It trades grasp
# accuracy for estimate accuracy, and the accuracy that matters at the end is the
# commit's.
#
# If depth really is marginal at 200 mm (todo D4, still unmeasured on the cell),
# fix the COMMIT rather than the standoff -- shorten it, or close the loop lower
# -- because moving the hover out makes the pick worse.
SERVO_STANDOFF = 0.35

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
# means a completed grasp. These are the gripper's own registers -- jaw position
# and actuator force -- so nothing here depends on knowing where the object is.
#
# The RG2 is a four-bar linkage driven through ONE knuckle joint, so there is no
# per-side driver pair to name any more; MujocoBackend reads whichever joint the
# model exposes. GRASP_DRIVER_JOINTS is retained only for the older proxy and is
# no longer used by the harness.
GRASP_DRIVER_JOINTS = ('gripper/right_driver_joint', 'gripper/left_driver_joint')
# Measured on this model: closing on nothing drives the driver joints to 0.77
# (range 0..0.8) with the actuator force at -0.13 as the linkage bottoms out.
# Closing on the 60 mm cube (OBJECT_SIZE above: half-extents of 0.03) stalls
# them at 0.25 with the force saturated at +5.0 and the finger velocity at ~0.
GRASP_CLOSED_Q = 0.60         # above this the jaws plainly closed on nothing
GRASP_FORCE_MIN = 1.0         # force needed to call it gripped, not just touched
GRASP_VEL_TOL = 0.02          # rad/s; below this the fingers count as stalled
GRASP_LIFT_RECHECK = 0.04     # m; lift clear and re-read after a failed check

# ---- gripper: OnRobot RG2 (hardware) --------------------------------------
# The pickup on hardware is OPEN LOOP: command the jaws, wait, lift. The RG2
# takes force as a WRITE-ONLY input and reports only jaw opening in mm, so
# there is no force to test against and the simulation's GRASP_FORCE_MIN above
# does not apply (see notes/RG2-GRIPPER-XMLRPC.txt).
#
# OPERATIVE constant: seconds for one full open->close stroke. MEASURE it at
# the bench with a stopwatch. None means unmeasured, so nothing can silently
# treat it as a real timing. STEPS_CLOSE/STEPS_RELEASE are control cycles and
# do not describe hardware time (24 s / 16 s at 125 Hz, far slower than real).
GRIP_STROKE_TIME_S = None
#
# OPTIONAL instrumentation, NOT a gate. If get_rg_width() answers, these let
# the pickup report where the jaws actually finished -- useful for the report
# and for diagnosing a missed pick -- but no decision branches on them.
# Closing on the 60 mm cube leaves the jaws near the cube side; closing on air
# drives them to the inner limit near zero, so the two are far apart.
OBJECT_SIDE_MM = OBJECT_HEIGHT * 1e3         # 60.0 -- derived from OBJECT_SIZE
GRASP_WIDTH_MARGIN_MM = 10.0                 # jaw padding, slop, cube tolerance
GRASP_WIDTH_MIN_MM = OBJECT_SIDE_MM - GRASP_WIDTH_MARGIN_MM   # 50.0

# ---- gripper: RG2 interface ----------------------------------------------
# PRIMARY control AND readback: the OnRobot URCap's XML-RPC on port 41414,
# through the `onRobot` package -- the method the lab example uses
# (UR3_RG2_rosbridge/examples/15_real_gripper_control.py):
#       RG(host, id).rg_grip(width_mm, force_N)     .get_rg_width()
# It is proportional (target width in mm) and it is the only route that reports
# width at all.
#
# LAST RESORT: trigger UR programs that were built on the pendant, over the
# dashboard server. Discrete open/close only, no readback, and it carries three
# quirks (report 5.2). Used automatically only when XML-RPC is unavailable.
UR_DASHBOARD_PORT = 29999
RG2_XMLRPC_PORT = 41414
RG2_OPEN_PROGRAM = '/programs/rg_open.urp'
RG2_CLOSE_PROGRAM = '/programs/rg_close.urp'
# The pipeline commands grip as GRIP_OPEN..GRIP_CLOSED (0..255); the RG2 wants
# a target width in mm, so the command is interpolated between these.
RG2_OPEN_WIDTH_MM = 110.0    # RG2 full stroke, from the datasheet
RG2_CLOSE_WIDTH_MM = 10.0    # fully closed without slamming; the example uses 10
RG2_GRIP_FORCE_N = 20.0      # write-only input; never readable
RG2_MIN_STEP_MM = 2.0        # skip commands smaller than this
# Width -> the simulation's driver-joint convention. The sim reads HIGH q as
# CLOSED (closing on air drives its joints to 0.77), while the RG2 reports jaw
# opening in mm, so the map has to invert: wider jaws -> smaller q. Keeping the
# direction means no existing threshold silently flips meaning.
GRIP_WIDTH_FULL_MM = RG2_OPEN_WIDTH_MM
GRIP_Q_FULL = 0.8             # sim driver-joint range, for the monotone map
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

# ---- hardware: UR3e + Robotiq 2F-85 + D435i -------------------------------
#
# Everything here is a property of the CELL, not of the simulator. The link
# data is shipped separately, in models/ur3e/arm_dynamics.json, so the robot
# build needs no MuJoCo.
#
# The values marked MEASURE are the ones that must be checked on the real cell
# before the accuracy means anything; tests/test_arm_control.py is the
# checklist.

UR_IP = '192.168.0.194'       # controller address
UR_CYCLE_DT = 0.008          # s, servoj period (125 Hz control loop)
UR_LOOKAHEAD_TIME = 0.1      # s, servoj lookahead
UR_SERVO_GAIN = 300          # servoj gain (controller-side)
UR_GRIPPER_UNIT = 9          # Modbus unit id of the 2F-85
UR_TRACK_CHUNK = 250         # servo iterations per operator-paced track step

# Identified servo gains for the observer/linearisation.
# PLACEHOLDER: these are the SIMULATED actuator's numbers, not the real
# controller's. Identify them from a step response (see --real) before trusting
# anything derived from A_cl.
UR_SERVO_KP = 1000.0
UR_SERVO_KV = 200.0

# Home configuration. MEASURED: the arm hand-guided to its true home on the real
# cell, read back with getActualQ() on 2026-10-23. This no longer matches
# models/ur3e/ur3e.xml's 'home' keyframe -- the simulated keyframe is now the
# OLD home. Joint 1 is +1.571, not -0.366: the base is swung ~111 deg round from
# the model, and the table was moved to match (see TABLE_* above).
UR_HOME_Q = (1.5708, -1.6299, 0.9853, -0.9276, -1.5708, 0.0)

# Mount: wrist_3_link -> gripper base (the frame the robot controls).
# MEASURE on the gripper; these are the composed model's values.
TOOL_MOUNT_POS = (1.0e-06, -3.0e-06, -1.08e-02)
TOOL_MOUNT_QUAT = (0.7071068, -0.0001, 0.0002, 0.7071068)

# Root yaw: the model's base frame -> the CELL's base frame.
# MEASURED (2026-10-23, Test 1). Five hand-guided poses, FK vs getActualTCPPose:
# the chain came out at -(TCP.x, TCP.y) every time, z unchanged, so the real
# cell is turned 180 deg about the base z axis from models/ur3e. This is the
# same swing that showed up in UR_HOME_Q (joint 1 = +1.571, not -0.366) and put
# the table at -y. The link lengths were already right; only the heading was not.
UR_ROOT_YAW = 3.141592653589793

# R_DOWN is defined further up, in the MODEL frame, because that is where it was
# authored. Everything downstream -- servo.desired_pose, pick, solve_ik -- targets
# the CELL frame, and the model was rotated onto that frame by UR_ROOT_YAW. Left
# as written, R_DOWN put the tool's +x on world +x while UR_HOME_Q has it on
# world -x, so the first orientation command spun the wrist 180 deg and swung the
# off-axis camera across the arm.
#
# Re-expressed here, once, so there is a single convention: this constant now
# agrees with the rotated model and with UR_HOME_Q. With UR_ROOT_YAW = pi,
# [[1,0,0],[0,-1,0],[0,0,-1]] becomes [[-1,0,0],[0,1,0],[0,0,-1]] -- the approach
# still points down, the azimuth flips by half a turn.
_c_yaw, _s_yaw = np.cos(UR_ROOT_YAW), np.sin(UR_ROOT_YAW)
R_DOWN = np.array([[_c_yaw, -_s_yaw, 0.0],
                   [_s_yaw, _c_yaw, 0.0],
                   [0.0, 0.0, 1.0]]) @ R_DOWN

# ---- camera mount: WRIST (eye-in-hand) ------------------------------------
# Transform gripper base -> D435i optical frame. The printed clamp goes on the
# RG2 PALM, so the camera will NOT sit where the simulator's bracket did. The
# numbers below ARE that bracket, kept so imports work: they are NOT measured.
#
# CAM_MOUNT_MEASURED was a boolean with no consumers; removed (todo F8). The
# status it encoded is now the difference between the shared pose and a real
# measurement -- i.e. whether HW_CAM_DELTA below is still identity.
#
# Sanity bounds for the measured values: expect the camera body roughly 40-100
# mm from the gripper base, optical axis near -z, quaternion components inside
# [-1, 1] and near unit norm. A translation in the hundreds of mm, or a
# non-unit quaternion, means a frame or units error, not a bad calibration.
#
# ONE POSE, TWO CONSUMERS (todo F7). The simulator's depth_cam and the hardware
# path use the SAME numbers below; there is no separate sim bracket any more.
# Enter them by hand once measured. The real camera will still not land exactly
# there, so the hardware path composes HW_CAM_DELTA on top -- that delta is where
# measured calibration offsets are recorded, and identity means "not measured".
CAM_MOUNT_POS = CAM_POS
CAM_MOUNT_QUAT = CAM_QUAT

HW_CAM_DELTA_POS = (0.0, 0.0, 0.0)             # hardware-only, measured (D5)
HW_CAM_DELTA_QUAT = (1.0, 0.0, 0.0, 0.0)       # wxyz, identity = unmeasured

# ---- camera mode -----------------------------------------------------------
# 'wrist'  eye-in-hand: the camera is bolted to the gripper, so its pose is
#          FK(q) @ CAM_MOUNT and it moves with the arm.
# 'fixed'  eye-to-hand: the camera is on a stand or the cell frame, looking at
#          the table. Its pose in the base frame is constant and comes from the
#          three values below, not from the arm.
CAMERA_MODE = 'wrist'

# Fixed camera (the fallback): where it sits and what it looks at, in the ROBOT
# BASE frame. MEASURE both with a tape measure and a target on the table, then
# refine by comparing a detected cube against a ruler. FIXED_CAM_UP only sets
# the roll.
# The target is DERIVED from TABLE_CENTRE. It used to be (0.275, 0.035, 0.02),
# the OLD simulated table -- which now points at where the table is not.
# FIXED_CAM_MEASURED was a boolean with no consumers; removed (todo F8). Whether
# this camera has been measured is recorded by the values themselves.
FIXED_CAM_POS = (0.0, -0.35, 0.75)                             # PLACEHOLDER
FIXED_CAM_TARGET = (TABLE_CENTRE[0], TABLE_CENTRE[1], TABLE_HEIGHT)
FIXED_CAM_UP = (0.0, 0.0, 1.0)
