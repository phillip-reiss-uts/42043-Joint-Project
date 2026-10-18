# Vision-Guided Pick (UR3e + 2F-85 + D435i)

MuJoCo simulation of a vision-guided pick task on a UR3e with a wrist-mounted
RealSense D435i RGB-D camera and a Robotiq 2F-85 gripper.

The robot can be commanded to **track** a moving cube with its camera, and to
**pick it up** or **put it back down** on demand.

## Target hardware

The simulated platform is the real one: **UR3e + RealSense D435i + Robotiq
2F-85**. The arm is vendored in `models/ur3e/` (menagerie has no UR3e) and the
gripper comes from `mujoco_menagerie`. Set the arm with `ROBOT_ARM` in
`src/config.py`.

Nothing in the robot layer depends on which arm is loaded — the geometry comes
through `src/kinematics.py` and the clock through `src/robot_backend.py` — so
swapping arms is a config change plus a matching `Kinematics` provider.

---

# Architecture

Robot code must run on real hardware; the object side is a simulation
demonstration. Three seams keep them apart, each enforced by a test.

## 1. The robot / world boundary — `src/world.py`

    robot side (hardware-portable)      world side (simulation)
    ------------------------------      ----------------------
    task.py       (WAIT/TRACK/PICK/…)   world.py        (boundary interface)
    arm.py        (joints, gripper)     pick_place.py   (scene, spawn, pool)
    servo.py      (PBVS)                target_motion.py (cube glide, sim only)
    controller.py (PD / LQR)            camera.py       (RGB-D sensor)
    observer.py   (EKF)                 model_loader.py (scene composition)
    state_space.py (linearisation)      overlay.py      (viewer overlays)
    tracking.py   (Kalman filter)       run_simulation.py (sim front-end)

The robot reaches the world **only** through `ObjectWorld`, which exposes
read-only methods: `perceive()` returning `TargetView` records (camera-derived
estimates, never ground truth) and `gripper_pose()` (the robot's own pose).
There is deliberately no method for moving or spawning an object.

`tests/test_layering.py` enforces this mechanically:

    python tests/test_layering.py

It fails if a robot module imports `target_motion`, `pick_place`,
`model_loader`, `camera`, `calibration` or `overlay`, or touches object state.
`task.py` is the strictest case: **no `mujoco` at all**, so the task policy is
expressed independently of any simulator.

## 2. The kinematics seam — `src/kinematics.py`

FK, the Jacobian, `M(q)`, `g(q)` and the servo gains all come from a
`Kinematics` provider rather than from direct MuJoCo calls:

    robot layer                          provider
    -----------                          --------
    arm.solve_ik        pose_and_jacobian
    servo.joint_step    jacobian          MujocoKinematics  (today)
    observer            mass_gravity, servo_gains
    state_space         mass_matrix, gravity_torque, servo_gains
                                          URKinematics      (on hardware)

The layering test fails if any robot module calls `mj_kinematics`,
`mj_jacBody`, `mj_fullM`, `mj_crb`, `mj_rne`, `mj_factorM` or `mj_comPos`
directly. `mj_step` is exempt — it is the clock question, not a kinematics one.

### M(q), g(q) and the servo gains

A real controller does not hand you an inertia matrix, so both are **modelled**.

`src/arm_dynamics.py` computes `M(q)` and `g(q)` from link data with a composite
rigid-body formulation. `AnalyticKinematics` wraps it. Verified to machine
precision against MuJoCo:

    python tests/test_dynamics.py
    worst |dM|  9.992e-16      worst |dg|  8.882e-15
    |M - M^T|   0.000e+00      min eigenvalue  0.100934 (positive definite)

Three things the URDF does not make obvious, all handled:

- **Rotor inertia.** `dof_armature` is non-zero on every joint; omitting it is a
  large error against wrist link inertias. It is a *required* constructor
  argument so a caller cannot forget it.
- **The gripper's internal joints.** The 2F-85 has finger joints of its own, so
  the distal subtree cannot be merged by "stop at the first jointed body". The
  whole subtree is folded into `wrist_3` as one rigid body.
- **The `base` body's 180° rotation**, which every downstream frame inherits.

## 3. The robot backend seam — `src/robot_backend.py`

**Who owns the clock**, and how joint state is read and written. `arm.py` and
`servo.py` contain no `import mujoco` at all. `RobotBackend` exposes:

    joint_positions() / joint_velocities() / time() / dt
    command(q_arm, grip)      # set targets WITHOUT advancing time
    step()                    # advance one control cycle
    gripper_state()           # {'q', 'v', 'force'} -- the Modbus trio
    joint_addresses(name)     # model handles

`MujocoBackend.command()` also applies **gravity feedforward** — see
*Gravity compensation* below. A `URBackend` would map `command`/`step` onto RTDE
`servoj` and `gripper_state` onto 2F-85 Modbus registers.

The residual gap is **timing, not logic**: a real `step()` blocks for the
control period and returns noisy encoders, whereas the simulated backend steps
instantly and reads perfect state. That is what `ArmObserver` exists to absorb.

---

# Control

## The task state machine — `src/task.py`

    WAIT    idle at home, nothing on the table
    TRACK   an object is present: follow it at standoff height, never descend
    PICK    an explicit pickup command is running
    HOLD    the cube is in the jaws, waiting for the next command
    PUTDOWN setting a held cube back on the table

The pickup key is a **toggle on whether a cube is held**:

| press | mode after | holding | action |
|---|---|---|---|
| 1 | HOLD | yes | grasp |
| 2 | TRACK | no | put the cube back down |
| 3 | HOLD | yes | grasp again |

A put-down runs to completion before the mode changes, so a third press cannot
interleave with the placement.

## The visual servo — `src/servo.py`

Position-based visual servoing with a damped-least-squares law:

- `kp * err_p` and `ko * err_o` form the commanded twist.
- **Translation and rotation are rate-limited separately.** A single limit taken
  from the position error alone leaves the angular rate unbounded; with a small
  position error and a large yaw error that swung the tool up and over.
- **Joint-space damping** near the target (default off, see below).
- **Integral term** on the pose error (default off, see below).
- The commanded joint vector **persists across `servo()` calls**, so calling the
  loop in chunks gives the same motion as one long call. This matters for the
  viewer, which must render between chunks.
- A constant-velocity Kalman filter (`src/tracking.py`) smooths the vision
  stream and supplies a velocity for the lead term.
- Occlusion recovery: when the jaws begin to clip the object the measured span
  drops, and the servo stops trusting the estimate.

### The grasp rules

Unchanged and deliberate:

- **Track until the target has been still for 3 s** (`SERVO_STILL_TIME`) before
  committing.
- **Abort the descent if the target moves** during it.
- **Verify the grasp** from the gripper's own position and force rather than
  assuming success from the servo converging, with a lift-and-recheck when the
  first reading is empty.
- **On a failed grip**, re-home and search again.

## Gravity compensation — `src/robot_backend.py`

A MuJoCo `<position>` servo produces `kp*(ctrl - q) - kv*qvel` and **nothing at
zero error**. Gravity still acts, so a commanded hold accelerates; the error
then builds and the arm oscillates and walks away. `MujocoBackend.command()`
therefore feeds `g(q)` forward onto `qfrc_applied`.

Two details that matter:

- **Only `g(q)`, not the full `qfrc_bias`.** `qfrc_bias` is `C(q,qdot) + g(q)`,
  and feeding Coriolis back is circular — it depends on the velocity the servo
  is nulling, so it pumps energy in and the arm never settles.
- The evaluation runs on a **scratch `MjData` with `qvel` zeroed**, so the live
  state is never mutated. Writing to `model.opt.gravity` to isolate the term
  corrupts the run.

Measured holding a fixed pose:

| model | without comp | with comp |
|---|---|---|
| bare arm (`ur3e.xml`, 6 actuators) | 1.79 rad | **0.000000 rad** |
| composed (arm + gripper, what runs) | 0.0127 rad | **0.0198 rad** |

The bare-arm figure is the clean demonstration that the mechanism is right, but
note the composed model — the one the simulation actually runs — still carries
~0.02 rad (about 14 mm at the tool) and the compensation does not currently
reduce it. That residual is a known open issue, not a solved one.

## Turning the integral and damping on

Both default to **off**, and should stay off in simulation. They were added
while chasing a sag-and-jerk that turned out to be a bug in `servo()` (the
commanded joint vector was being re-seeded from `data.qpos` on every call, so
chunking ratcheted the command downhill). With that fixed, both measurably
*hurt*:

    ki    damp_gain   reversals   qvel_sd   z_err
    0.0   0.0         9           0.0023    0.05 mm   <- used
    0.5   0.0         4           0.0034    0.16 mm
    0.0   1.0         9           0.0038    0.19 mm
    0.5   1.0         5           0.0043    0.17 mm

They are kept because on hardware the gravity feedforward is *approximate*
rather than exact, and a standing bias is exactly what the I term is for.
Re-enable and re-tune then; **do not assume these numbers transfer.**

---

# Setup

    python -m pip install -r requirements.txt

Verified with Python 3.12.10, mujoco 3.10.0, numpy 2.5.2, scipy 1.18.1,
matplotlib 3.11.1 on macOS 13.7.8 (x86_64).

`pyrealsense2` is **not** in `requirements.txt`: there is no wheel for Python
3.12 on macOS. `src/world_real.py` imports it lazily and falls back to
`FakeDepthSource`, so everything except a live camera works without it.

# Run

    python main.py                # simulation (current default)
    python main.py --sim          # simulation, explicit
    python main.py --hardware     # real robot (not implemented; exits 2)

`main.py` is a minimal dispatcher: it chooses a front-end and nothing else. With
no option it runs the simulation. Once the hardware front-end exists the
sensible default becomes the real robot and simulation should require `--sim`;
that is a one-line change in `main.py`'s `_parse`.

On macOS, use `mjpython` for the viewer:

    mjpython main.py

The front-end can also be run directly:

    python run_simulation.py

## Usage

- `6` — spawn the cube on the table.
- `7` — **pick up / put down**. First press grasps, second sets the cube back
  down on the table, third grasps again.
- `8` — pause / resume. The robot holds position and resumes where it left off.
- `9` — reset the scene.
- `.` — cycle the object's motion band: off / slow 15 / medium 30 / fast 45 mm/s.

Keys are `6`–`9` and `.` because the MuJoCo viewer binds every letter `a`–`z` to
one of its 26 render flags and digits `0`–`5` to geom groups, so nothing else is
free.

## Tests and harness

    python tests/test_layering.py            # robot/world separation
    python tests/test_dynamics.py            # analytic M(q)/g(q) vs MuJoCo
    python tests/test_hardware.py --sim      # hardware assumptions, sim mode
    python tests/test_hardware.py --real     # acceptance checklist, real mode
    python tests/test_hardware.py --selfcheck # proves the checks can fail
    python run_tests.py --trials 8 --band medium --kf both

`test_hardware.py --sim` currently reports **28 passed, 0 failed**.
`--real` reports 21 skipped and exits 1: it is an acceptance checklist, not a
pass, and says so.

`run_tests.py` writes `summary.csv` and four graphs.

### Demonstrations

Each needs to compose the scene, so they live outside the robot modules:

    python -m src.demo_servo        # PBVS vs open-loop planning
    python -m src.demo_controller   # PD vs LQR outer loop
    python -m src.demo_observer     # velocity estimation from encoders
    python -m src.demo_state_space  # controllability / observability

---

# The workcell

There is **one flat 400 mm square table and nothing else**. An earlier design
had a chute cut into the table's far edge for dropping objects into; it was
removed — a real cell has no such bin, and as placed it sat inside the table
footprint with half its scatter radius off the edge. A picked object is simply
set back down on the tabletop.

The geometry is tuned to the UR3e's ~500 mm reach, not the UR5e's ~850 mm:

    TABLE_CENTRE   [0.275, 0.035]     TABLE_HALF  [0.20, 0.20]
    SPAWN_X_RANGE  0.240 .. 0.385
    SPAWN_Y_RANGE  -0.075 .. 0.145

The spawn region is the intersection of what the arm can **reach** and what the
wrist camera can **see** from the home pose. Those stopped coinciding once the
table was widened: the camera sits on the flange and looks at the table centre,
so an object near the near edge (x < ~0.24) is invisible from home even though
the arm could reach it. Objects released by a put-down are clamped into this
region for the same reason.

`models/ur3e/ur3e.xml` carries a `home` keyframe hovering over the table centre
with the tool pointing straight down, and an `attachment_site` site at the
**tool0** frame (identity with `wrist_3_link`, *not* the `flange` frame, which
points the tool sideways).

---

# Layout

- `main.py` — minimal entry point: dispatches to a front-end.
- `run_simulation.py` — **simulation front-end**: passive viewer, keyboard,
  scene reset, the moving cube.
- `run_tests.py` — headless trial harness; writes `summary.csv` and four graphs.
- `tests/test_layering.py` — enforces the robot / world separation.
- `tests/test_hardware.py` — validates hardware assumptions (sim + real modes).
- `tests/test_dynamics.py` — analytic `M(q)`/`g(q)` vs the simulator.
- `models/ur3e/` — vendored UR3e MJCF and STL meshes.
- `src/world.py` — **the boundary**: `TargetView`, `ObjectWorld`, `SimObjectWorld`.
- `src/task.py` — WAIT/TRACK/PICK/HOLD/PUTDOWN state machine. Pure policy: no
  mujoco, no viewer, no scene.
- `src/arm.py` — IK, trajectory execution, gripper, grasp readback.
- `src/calibration.py` — camera intrinsic calibration from rendered marker views.
- `src/camera.py` — RealSense D435i wrist camera: capture and deprojection.
- `src/config.py` — global flags and constants.
- `src/controller.py` — joint and LQR controllers, encoder model.
- `src/model_loader.py` — composes arm + gripper + wrist camera + table.
- `src/kinematics.py` — **the geometry seam**: FK/Jacobian/M/g providers.
- `src/robot_backend.py` — **the I/O seam**: stepping, joint read/write,
  gripper, gravity feedforward.
- `src/arm_dynamics.py` — Newton-Euler `M(q)`/`g(q)` from link data.
- `src/world_real.py` — **hardware world**: RealSense depth behind `ObjectWorld`.
- `src/observer.py` — observer for the unmeasured joint states.
- `src/overlay.py` — live viewer overlays.
- `src/perception.py` — clustering, convex hull, calipers, object detection.
- `src/pick_place.py` — scene, spawn, one pick, put-down, table clearing.
- `src/servo.py` — PBVS servo, occlusion recovery, open-loop grasp commit.
- `src/state_space.py` — linearisation, controllability/observability, LQR design.
- `src/target_motion.py` — moving-target driver and speed bands (sim only).
- `src/tracking.py` — constant-velocity Kalman filter on the vision estimate.
- `requirements.txt` — dependencies.

---

# Status

## Working

- Tracking: the arm follows a moving cube at 0.20 m standoff with the tool held
  level. Hover xy error is **0.4–3.0 mm** against ground truth.
- Detection: the wrist camera locates the cube to ~1.3 mm.
- Gravity compensation: removes **1.79 rad** of hold drift on the bare arm
  model. The composed model still shows ~0.02 rad; see *Gravity compensation*.
- Put-down: the pickup key toggles grasp → release → grasp, with the cube set
  back down gently at its own xy.
- The grasp is verified from the gripper's own registers, not assumed.
- All three test suites pass.

## Known limitation: grasp accuracy

**5 of 10** grasps succeed in the current harness. The failures are a lateral
offset introduced during the descent, not a perception error:

    at hover       1.5 mm mean xy error
    at grasp       5.7 mm mean xy error

A ~17 mm lateral offset on a 30 mm cube puts the pads against the cube's side on
the way down; they deflect and close on air (`jaw_q` reads 0.785, fully closed,
in every failure).

The descent currently interpolates the **joint** vector to the IK solution. A
straight line in joint space is not a straight line for the tool — as the arm
folds down the tool swings sideways. The mechanism is measured: the commanded z
falls 0.377 → 0.156 while the tool only reaches 0.287, and the xy error grows
monotonically with that lag (0.9 → 3.0 → 10.1 → 16.7 → 20.3 mm).

Three attempts at fixing this (a Cartesian straight-line descent, per-target
settle steps, and a closed-loop settle) did not reduce the error and were
**reverted**. Diagnosing why the arm lags a ramp it should track is the next
step; it is the same shape as the `_q_cmd` bug that was fixed.

## Also open

- A fully hidden object is still not found by the recovery search.
- `SERVO_COMMIT_TRIES` bounds the abandon-and-retrack cycles; a target that
  never holds still exhausts them and the pick is reported as failed.

---

# Porting to hardware

## Done: `src/world_real.py`

`RealObjectWorld` implements the boundary against a RealSense depth stream.
`pyrealsense2` is imported lazily, so the module imports cleanly without
librealsense, and `FakeDepthSource` exercises the whole
depth → deproject → detect path. The three things the simulator gave for free
are now explicit:

- **Extrinsics** — camera pose is `FK(q) @ T_flange_cam`, so the mount offset
  must be measured. A few millimetres is a direct bias on every deprojected
  centre.
- **Intrinsics** — read from the device, so `fx != fy` and an off-centre
  principal point are handled rather than assumed away. `check_intrinsics()`
  reports the difference instead of asserting equality.
- **Depth conventions** — `uint16` millimetres scaled to metres, converted
  explicitly; MuJoCo's metres/z-forward convention is not assumed.

**`RealSenseSource` has never been executed** — there is no `pyrealsense2`
wheel for Python 3.12 on macOS. It is written against the API but is unverified
until it runs against a device.

## Still to do

- **`URBackend`** — RTDE `servoj`/`movej` for `step()`, Modbus for the 2F-85
  registers `Arm.grasp_state` already consumes.
- **`URKinematics`** — FK/Jacobian from the controller, with `arm_dynamics`
  supplying `M(q)`/`g(q)`.
- **Camera mount + table plane** — measure `T_flange_cam` and `TABLE_HEIGHT`.
- **Rotor inertia and `Kp/Kv`** — identify both; `tests/test_hardware.py --real`
  describes the procedure for each. The invariant to measure is the **time
  constant** `Kp/Kv`, not the damping ratio: `ζ` varies with pose through
  `M_jj`, whereas `Kp/Kv` is fixed by the servo and is what the observer's
  prediction was tuned against.
- **Re-tune the integral and damping** — see above; the simulation numbers do
  not transfer, because on hardware the gravity feedforward is approximate.

Task logic (`src/task.py`), the servo law and its occlusion recovery, the EKF,
the LQR design and the grasp verification need no change.
