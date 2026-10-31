import math
import numpy as np


# ---- workcell -------------------------------------------------------------
# UR3e. Table is axis-aligned: 440 mm along x, 300 mm deep in -y (measured
# 2026-10-23). The tabletop is 15 mm ABOVE the robot base plane (measured).
TABLE_HEIGHT = 0.015
TABLE_THICK = 0.02
TABLE_CENTRE = [0.0, -0.270]
TABLE_HALF = [0.22, 0.15]

# Valid-object region for the depth->world mask: (cx, cy, hx, hy). Only the
# tabletop can hold a graspable object.
TABLE_XY_BOUNDS = (TABLE_CENTRE[0], TABLE_CENTRE[1], TABLE_HALF[0], TABLE_HALF[1])

OBJECT_SIZE = [0.03, 0.03, 0.03]          # HALF-extents
OBJECT_HEIGHT = 2 * OBJECT_SIZE[2]        # full height, 60 mm
OBJECT_MASS_KG = 0.090                    # weighed
# Sliding friction; MuJoCo takes the max of the two in a contact. Explicit
# because both used to fall back on defaults nobody chose.
GRIP_PAD_FRICTION = 2.5
OBJECT_FRICTION = 1.5

# ---- RG2 ----------------------------------------------------------------
# Full-stroke arc, measured on the tool: tips sweep 55 mm, drop 39.2 mm, stroke
# 110 mm. Reference only.
RG2_MAX_OPENING = 0.110
RG2_TIP_DROP_FULL = 0.0392
RG2_FINGER_RADIUS = 0.055

# ---- pick height ---------------------------------------------------------
# Derived in the grasp / place section below, once TABLE_HEIGHT exists.

# ---- object pool ---------------------------------------------------------
POOL_SIZE = 1
INITIAL_OBJECTS = 0                       # spawned on scene reset
MIN_OBJECT_SEPARATION = 0.10              # centre-to-centre

# Where a released object is put down, relative to the table, and the parking
# pose objects are frozen at until spawned.
PLACE_EDGE_MARGIN = 0.10
PARK_POS = [5.0, 5.0, -0.6]

# ---- camera --------------------------------------------------------------
# TOOL FLANGE -> D435i optical frame (flange, not gripper base). Measured pair
# on the cell: tool0 300 mm and lens 186 mm above the base plane, i.e. the lens
# is 114 mm below the flange, so CAM_POS_z = 0.114 m.
CAM_POS = (0.046, -0.0080, 0.114)
CAM_QUAT = (0.008026, -0.695654, -0.718261, -0.010057)
MESH_QUAT = [0.707, 0.0, 0.0, 0.707]
MESH_POS = CAM_POS

# Printed bracket mass, modelled as a point mass at the midpoint between palm
# and lens. GUESS for PLA, not a measurement; 0.0 models the camera alone.
# Rebuild with tools/build_arm_dynamics.py after changing it.
CAM_BRACKET_MASS_KG = 0.030

# ---- arm / gripper identity ----------------------------------------------
ROBOT_ARM = 'ur3e'                        # models/ur3e/ur3e.xml
ROBOT_GRIPPER = 'onrobot_rg2'             # models/onrobot_rg2/onrobot_rg2.xml
ROBOT_CAMERA = 'realsense_d435i'

ARM_DOF = 6
# Per-joint IK clip, rad. The UR3e allows +-2*pi; the MJCF models the elbow as
# +-pi, so this is the range both can hold.
JOINT_RANGE = (2.0 * math.pi,) * ARM_DOF
GRIP_OPEN = 0.0
GRIP_CLOSED = 255.0
# RG2 linkage: finger_width 0 = SHUT, opposite to the command space.
GRIP_FINGER_OPEN_ANGLE = -0.35692
GRIP_FINGER_SHUT_ANGLE = 0.785398
GRIP_FINGER_MIMIC_GAIN = 10.38471         # rad of knuckle per metre of width
# Tool pointing down, in the MODEL frame; rotated onto the cell frame below.
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])

# ---- motion rates (CNC style) --------------------------------------------
RAPID = 0.10
FEED = 0.05
RAPID_MIN_STEPS = 200
FEED_MIN_STEPS = 400

# ---- contact guard for the descent ---------------------------------------
# gripper base -> lowest pad, jaws OPEN (the descent runs open). The descent
# may never command the pads nearer the table than the clearance, so a wrong
# grasp height stops the arm instead of driving through the table.
GRIPPER_TO_PAD = 0.2207
# wrist_3_link -> gripper base. +z points DOWN the approach at R_DOWN, so the
# flange->gripper-top distance is POSITIVE. UNMEASURED; needed by the MODEL and
# by the gripper-base conversion, not by calibration.
TOOL_MOUNT_POS = (1.0e-06, -3.0e-06, 2.30e-02)
TABLE_CONTACT_CLEARANCE = 0.002
# Descents ending within the slow zone are run at the slow factor.
CONTACT_SLOW_ZONE = 0.030
CONTACT_SLOW_FACTOR = 0.3

# Viewer frame pacing: sync once per ~16 ms of simulated time (~60 Hz).
SYNC_DT = 0.016
SYNC_STEPS = 8                            # steps between syncs

# ---- perception ----------------------------------------------------------
SELF_FILTER = 0.20
SURFACE_BAND = 0.005
DEPTH_MAX = 5.0
# Lowest height above the tabletop that counts as object. Raised above the
# ~20-30 mm a flat hand sits at, so a hand cannot outcompete the cube; too high
# and every detection disappears. Does not move the grasp height.
HAND_FILTER = 0.015
# The one z floor both paths use: max() of the two independent rules above.
SURFACE_Z_MIN = TABLE_HEIGHT + max(SURFACE_BAND, HAND_FILTER)
MIN_OBJECT_POINTS = 1500
# Seconds between [PERC] console lines on hardware; 0 disables. A change in
# detection count always prints.
PERCEPTION_LOG_S = 0.5

# ---- state observer ------------------------------------------------------
ENCODER_NOISE = 1.0e-4                    # rad, measurement noise (R)
TAU_NOISE = 0.30                          # N.m, process noise
OBSERVER_RELIN = 100                      # steps between re-linearisations
OBSERVER_SUBSTEPS = 4                     # integration sub-steps/timestep

# Simulated calibration board geometry (built by model_loader).
CALIB_MARKERS = 7                         # markers per row and column (7x7)
CALIB_SPACING = 0.02                      # marker pitch, m
CALIB_MARKER_R = 0.003                    # marker radius, m
CALIB_BOARD_THICK = 0.004                 # slab thickness, m
CALIB_BOARD_MARGIN = 0.02                 # slab overhang beyond outer markers

# ---- feedback control on the estimated state -----------------------------
USE_OBSERVER = True                       # close the loop on the estimate
FB_KQ = 3.0                               # PD fallback position gain
FB_TD = 0.05                              # PD fallback velocity gain, s
USE_LQR = True                            # LQR gains instead of FB_KQ/FB_TD
LQR_Q_POS = 1.0                           # state cost, position error (rad^2)
LQR_Q_VEL = 1.0e-4                        # state cost, velocity error (rad/s)^2
LQR_R = 5.0e-2                            # command cost (rad^2); sets gain level

# ---- grasp / place -------------------------------------------------------
# ---- grasp / place -------------------------------------------------------
# The pick/place height: TOOL FLANGE above the tabletop, METRES. THIS IS THE
# ONE NUMBER TO EDIT. Every grasp, descent and put-down in both paths reads it.
PICK_Z_ABOVE_TABLE = 0.220
# World-space tool z for the grasp (flange frame, metres).
PICK_Z = TABLE_HEIGHT + PICK_Z_ABOVE_TABLE

# TIP_OFFSET is kept only as the geometric tool length: flange -> the cube's
# centre with the pads on the tabletop. Nothing derives a height from it.
GRIPPER_PAD_HEIGHT = 0.0298               # measured pad face; reference only
# Flange -> lowest pad, from the 242.3 mm measured with the pads on the table.
FLANGE_TO_PAD = GRIPPER_TO_PAD + TOOL_MOUNT_POS[2]
TIP_OFFSET = FLANGE_TO_PAD - OBJECT_HEIGHT / 2.0
APPROACH_Z = 0.10
FINAL_Z = 0.03
LIFT_Z = 0.15
PLACE_DX = 0.0
PLACE_DY = 0.0
PLACE_CLEARANCE = 0.002
STEPS_CLOSE = 3000
# Gripper command is quantised to this many levels: the URCap retargets on
# every distinct width, so a smooth ramp makes it stutter.
GRIP_RAMP_LEVELS = 8
STEPS_RELEASE = 2000
DROP_HOLD_STEPS = 500                     # hold open after a drop

# ---- spawn ---------------------------------------------------------------
RANDOMIZE_SPAWN = True
RANDOMIZE_YAW = True
SPAWN_CLEARANCE = 0.05                    # spawn height above the surface
SPAWN_MARGIN = 0.06                       # keep clear of the table edge
SPAWN_Y_MIN = None                        # UR5e-era footprint floor; off
SPAWN_ARM_CLEARANCE = 0.12                # xy clearance from gripper/base
SPAWN_X_MIN = None                        # re-measure on this cell first

_x_lo = TABLE_CENTRE[0] - TABLE_HALF[0] + SPAWN_MARGIN + OBJECT_SIZE[0]
_x_hi = TABLE_CENTRE[0] + TABLE_HALF[0] - SPAWN_MARGIN - OBJECT_SIZE[0]
_y_lo = TABLE_CENTRE[1] - TABLE_HALF[1] + SPAWN_MARGIN + OBJECT_SIZE[1]

if SPAWN_X_MIN is not None:
    _x_lo = max(_x_lo, SPAWN_X_MIN)
if _x_hi <= _x_lo:
    raise ValueError(
        'spawn x range is empty: lo %.3f >= hi %.3f. SPAWN_X_MIN and the table '
        'constants disagree about which workcell this is -- fix the config, do '
        'not clamp around it.' % (_x_lo, _x_hi))

SPAWN_X_RANGE = (_x_lo, _x_hi)
SPAWN_Y_RANGE = ((_y_lo if SPAWN_Y_MIN is None else max(_y_lo, SPAWN_Y_MIN)),
                 TABLE_CENTRE[1] + TABLE_HALF[1] - SPAWN_MARGIN - OBJECT_SIZE[1])

# Deliberately awkward spawns, for the perception-recovery path.
HARD_SPAWNS = (
    (0.180, -0.140),     # near the inside reach limit
    (0.320, 0.240),      # far y edge, marginal camera footprint
    (0.230, 0.000),      # close to the base, wrist near a singularity
    (0.390, -0.150),     # outer x limit
)
HARD_SPAWN_MODE = False
# Spawns in the blind strip nearest the base.
REALLY_HARD_MODE = False
HARD_BLIND_Y = (TABLE_CENTRE[1] - TABLE_HALF[1] + OBJECT_SIZE[1],
                TABLE_CENTRE[1] - TABLE_HALF[1] + 0.10)

# ---- search --------------------------------------------------------------
# Revolve the wrist when the nominal view cannot see the object.
REVOLVE_DEG = tuple(range(15, 260, 15))
REVOLVE_SPEED = RAPID                      # safe: gripper is high
SEARCH_SPEED = RAPID
# Offsets around the nominal gripper xy, tried in order.
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
DROP_RADIUS = 0.07                        # scatter around the drop centre
MAX_PASSES = 12                           # cap on clear_table passes

# ---- PBVS servo ----------------------------------------------------------
# Commanded twist caps, m/s. Transit is the hover leg (also the standoff the
# tracking servo holds), descent the final approach.
SERVO_V_TRANSIT = 0.080
SERVO_V_DESCENT = 0.015
SERVO_MAX_ITERS = 8000                    # per servo stage
SERVO_PERCEIVE_EVERY = 100                # iterations between observations

# Occlusion recovery: on a shrinking span, back up and commit open loop from
# the fullest-view estimate rather than servoing into the occlusion.
SERVO_OCCLUSION_RECOVERY = True
SERVO_OCCLUDE_RATIO = 0.95                # span below this fraction of best seen
SERVO_OCCLUDE_RECOVER = 0.97              # span must return above this to commit
SERVO_OCCLUDE_FAILS = 2                   # consecutive lost looks that trigger it
SERVO_RETREAT_STEP = 0.02                 # m per backup step
SERVO_RETREAT_MAX = 15                    # cap on backup steps
SERVO_RETREAT_V = 0.03                    # m/s for backup and open-loop commit

# Hover = PICK_Z lifted by this standoff, so the camera sits CAM_HOVER_HEIGHT
# above the table. Operator-set viewing choice. The tool-mount offset back to
# gripper base is applied in servo.desired_pose.
CAM_HOVER_HEIGHT = 0.380
SERVO_STANDOFF = CAM_HOVER_HEIGHT - CAM_POS[2] - PICK_Z + TABLE_HEIGHT

# Moving-target demo: bands cycled live with '.', index 0 is off. All below
# SERVO_V_TRANSIT so the arm can close the gap.
OBJ_MOVE_BANDS = (
    ('off',    0.000),
    ('slow',   0.015),
    ('medium', 0.030),
    ('fast',   0.045),
)
OBJ_MOVE_TURN_S = 1.2                     # s between random re-aims

# Vision Kalman filter: smooths the intermittent stream, bridges invisible
# stretches, supplies the velocity for the lead term.
SERVO_USE_KF = True
KF_SIGMA_MEAS = 1.5e-3                    # measurement noise, m
KF_SIGMA_ACC = 0.60                       # manoeuvre noise, m/s^2
KF_GATE = 3.0                             # innovation gate, sigmas; 0 disables
SERVO_LEAD_S = 0.15                       # s of lead ahead of the estimate

# PBVS proportional gains, 1/s. These set the speed; the caps only clamp.
SERVO_KP = 2.0
SERVO_KO = 2.0
# Integral + joint damping, both OFF: they measurably increased error. Kept
# for hardware, where the gravity feedforward is approximate.
SERVO_KI = 0.0
SERVO_KI_NEAR_M = 0.05
SERVO_KI_NEAR_RAD = 0.30
SERVO_KI_MAX_M = 0.02
SERVO_KI_MAX_RAD = 0.20
SERVO_DAMP_GAIN = 0.0                     # joint-space D
SERVO_DAMP_NEAR_M = 0.05
SERVO_DAMP_NEAR_RAD = 0.15
# Yaw blend gain, separate from the position blend: a square top face reads as
# yaw or yaw+90, and the raw label flips between looks.
SERVO_YAW_SMOOTH = 0.35

# ---- tracking a moving target --------------------------------------------
# Chase until the object has held still for the dwell, then commit open loop.
SERVO_TRACK_MOVING = True
SERVO_STILL_TIME = 3.0                    # s of observed stillness
SERVO_STILL_TOL = 0.008                   # m; filtered drift still "stationary"
SERVO_STILL_SPEED = 0.006                 # m/s; overlay label only
SERVO_TRACK_TOL = 0.020                   # m; also required to commit
SERVO_TRACK_TOL_ROT = 0.15                # rad
SERVO_TRACK_ITERS = 60000                 # iteration cap on the chase
SERVO_LOST_LIMIT = 5                      # consecutive failed looks before searching
SERVO_SEARCH_LIMIT = 2                    # fruitless searches before giving up
# Watch the target through the open-loop descent and abandon it if it moves.
SERVO_WATCH_DESCENT = True
# Perceived motion that counts as moving and aborts the descent. Raised from
# 15 mm, which view-dependent mount bias tripped on every descent.
SERVO_MOVE_TOL = 0.040
SERVO_COMMIT_TRIES = 5                    # abandon-and-retrack cycles

# ---- grasp verification (simulation) -------------------------------------
# Thresholds on the SIMULATED fingers: a real RG2 reports width only, no
# force, so the hardware pickup runs open loop on 'unknown'.
GRASP_DRIVER_JOINTS = ('gripper/right_driver_joint', 'gripper/left_driver_joint')
GRASP_CLOSED_Q = 0.60                     # above this: closed on nothing
GRASP_FORCE_MIN = 1.0                     # force to call it gripped
GRASP_VEL_TOL = 0.02                      # rad/s; below this: stalled
GRASP_LIFT_RECHECK = 0.04                 # m; lift clear and re-read

# ---- gripper: OnRobot RG2 (hardware) -------------------------------------
# Hardware pickup is OPEN LOOP: command the jaws, wait, lift.
# Seconds for one full open->close stroke. MEASURE it; None means unmeasured.
GRIP_STROKE_TIME_S = None
# Optional instrumentation, not a gate: where the jaws finished.
OBJECT_SIDE_MM = OBJECT_HEIGHT * 1e3      # 60.0
GRASP_WIDTH_MARGIN_MM = 10.0
GRASP_WIDTH_MIN_MM = OBJECT_SIDE_MM - GRASP_WIDTH_MARGIN_MM

# RG2 interface: URCap XML-RPC (proportional, reports width) first, dashboard
# programs (discrete, no readback) as last resort.
UR_DASHBOARD_PORT = 29999
RG2_XMLRPC_PORT = 41414
RG2_OPEN_PROGRAM = '/programs/rg_open.urp'
RG2_CLOSE_PROGRAM = '/programs/rg_close.urp'
# Grip is sent as GRIP_OPEN..GRIP_CLOSED (0..255); interpolate onto these widths.
RG2_OPEN_WIDTH_MM = 110.0                 # datasheet full stroke
RG2_CLOSE_WIDTH_MM = 56.0                 # jaws squeeze ~4 mm on the 60 mm cube
RG2_GRIP_FORCE_N = 2.5                    # write-only input; never readable
RG2_MIN_STEP_MM = 2.0                     # skip commands smaller than this

# Simulated actuator is on the finger hinge, so its range is a torque: pad
# force times the measured 47.6 mm lever.
GRIP_LEVER_MM = 47.6
GRIP_FORCE_RANGE = (-RG2_GRIP_FORCE_N * GRIP_LEVER_MM / 1000.0,
                     RG2_GRIP_FORCE_N * GRIP_LEVER_MM / 1000.0)

# Width -> sim driver joint. The sim reads HIGH q as CLOSED, so the map inverts.
GRIP_WIDTH_FULL_MM = RG2_OPEN_WIDTH_MM
GRIP_Q_FULL = 0.8

# ---- viewer --------------------------------------------------------------
VIEWER_REAL_TIME = True                   # pace the sim to wall-clock time
OVERLAY_CLOUD = True                      # live point-cloud panel
OVERLAY_CLOUD_EVERY = 30                  # viewer syncs between cloud refreshes

# ---- pick hang guard (simulation front-end) ------------------------------
# Wall-clock budget for one explicit pickup. A hang guard, not a performance
# target; tripping prints how long it took.
PICK_BUDGET_S = 300.0

# ---- debug flags ---------------------------------------------------------
VERIFY_IK = False                         # IK self-test at startup
VERBOSE = False                           # per-stage logging

# ---- hardware: UR3e + OnRobot RG2 + D435i --------------------------------
# Cell properties. Link data ships separately in models/ur3e/arm_dynamics.json.
UR_IP = '192.168.0.194'                   # controller address
UR_CYCLE_DT = 0.008                       # s, servoj period (125 Hz)
UR_LOOKAHEAD_TIME = 0.1                   # s, servoj lookahead
UR_SERVO_GAIN = 300                       # servoj gain
UR_TRACK_CHUNK = 250                      # servo iterations per track step

# Identified servo gains for the observer/linearisation. PLACEHOLDER: these are
# the simulated actuator's, not the real controller's.
UR_SERVO_KP = 1000.0
UR_SERVO_KV = 200.0

# Home on the cell, hand-guided and read back with getActualQ().
UR_HOME_Q = (1.5708, -1.6299, 0.9853, -0.9276, -1.5708, 0.0)
# Simulated home: centres the wrist camera on the table (VIEWING pose only;
# hardware homes to UR_HOME_Q).
SIM_HOME_Q = (1.2564, -1.6585, 1.0695, -0.9818, -1.5708, 1.2564)

# wrist_3_link -> gripper base ROTATION. The position is up by the contact
# guard, because the flange-to-pad length needs it.
TOOL_MOUNT_QUAT = (0.7071068, -0.0001, 0.0002, 0.7071068)

# Model base frame -> cell base frame: the cell is turned 180 deg about z
# (measured, FK vs getActualTCPPose).
UR_ROOT_YAW = 3.141592653589793
# R_DOWN is authored in the MODEL frame and used in the cell frame, so it is
# rotated once here. Single convention: agrees with the model and UR_HOME_Q.
_c_yaw, _s_yaw = np.cos(UR_ROOT_YAW), np.sin(UR_ROOT_YAW)
R_DOWN = np.array([[_c_yaw, -_s_yaw, 0.0],
                   [_s_yaw, _c_yaw, 0.0],
                   [0.0, 0.0, 1.0]]) @ R_DOWN

# ---- camera mount: wrist (eye-in-hand) -----------------------------------
# FLANGE -> camera. This is the reference everywhere on the hardware path.
# gripper base is CAM_MOUNT_POS minus TOOL_MOUNT_POS, and exists only because
# the simulated gripper's geometry is attached there.
CAM_MOUNT_POS = CAM_POS
CAM_MOUNT_QUAT = CAM_QUAT
# Same mount, in the gripper-base frame the MJCF builds the gripper in.
CAM_MOUNT_POS_GRIPPER = tuple(np.array(CAM_POS, float)
                              - np.array(TOOL_MOUNT_POS, float))
# Hardware-only measured correction on top of the above. Identity = unmeasured.
HW_CAM_DELTA_POS = (0.0, 0.0, 0.0)
HW_CAM_DELTA_QUAT = (1.0, 0.0, 0.0, 0.0)  # wxyz
