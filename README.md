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

Every seam is split the same way: an **interface** in a shared module that owns
no implementation, and an **implementation** in whichever layer runs it.

| interface (shared) | simulation implementation | hardware implementation |
|---|---|---|
| `src/kinematics.py` — `Kinematics` | `src/kinematics_mujoco.py` — `MujocoKinematics` | `src/kinematics_ur.py` — `URKinematics` |
| `src/robot_backend.py` — `RobotBackend` | `src/robot_backend_mujoco.py` — `MujocoBackend` | `src/robot_backend_ur.py` — `URBackend` |
| `src/world.py` — `ObjectWorld` | `src/world_sim.py` — `SimObjectWorld` | `src/world_real.py` — `RealObjectWorld` |

The entry point constructs the implementations and injects them; `arm.py` takes
its `Kinematics` and `RobotBackend` as required arguments and imports neither.
`tests/test_layering.py` fails if an interface module imports `mujoco` or one of
the simulation implementations, or if the robot layer reaches for an
implementation instead of being handed one.

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

## Gravity compensation — `src/robot_backend_mujoco.py`

A MuJoCo `<position>` servo produces `kp*(ctrl - q) - kv*qvel` and **nothing at
zero error**. A real UR3e controller carries the arm's weight internally; a
MuJoCo actuator does not. `MujocoBackend.command()` therefore feeds `g(q)`
forward onto `qfrc_applied` (via `gravity_feedforward`), and that is the **only**
compensation in the model.

The model itself must not compensate. `model_loader` used to set
`body.gravcomp = 1.0` on every body as well, which made the simulated arm
weightless and turned the backend's term into an upward disturbance. With both
active the arm sat off its commanded pose, and because the pads hang 112 mm
below the flange the tilt was amplified into lateral error at the grasp point:

| at a grasp pose | double (before) | single (after) |
|---|---|---|
| tool error | 7.3 mm | **2.1 mm** |
| tool tilt | 1.83° | **0.99°** |
| pad-centre error | 6.6 mm | **3.7 mm** |
| open-loop commit, pad error | 12.9 mm | **2.4 mm** |

Those figures were taken with the Euler integrator; the `implicitfast` setting
described below removes the remaining ~2 mm of hold error.

Two details that matter:

- **Only `g(q)`, not the full `qfrc_bias`.** `qfrc_bias` is `C(q,qdot) + g(q)`,
  and feeding Coriolis back is circular — it depends on the velocity the servo
  is nulling, so it pumps energy in and the arm never settles.
- The evaluation runs on a **scratch `MjData` with `qvel` zeroed**, so the live
  state is never mutated. Writing to `model.opt.gravity` to isolate the term
  corrupts the run.

The rule is the same on hardware: **each backend supplies whatever compensation
its controller needs, exactly once.** A `URBackend` streams joint targets over
RTDE and the UR controller does the compensating — it must not add a second
term. `tests/test_hardware.py --sim` now enforces this: `body.gravcomp` must be
zero and `qfrc_applied` must equal `g(q)` immediately after a command.

## Integration — `src/model_loader.py`

MuJoCo's default Euler integrator evaluates the actuator's velocity term
explicitly. At this model's 2 ms timestep that term is past the explicit
stability limit on the wrists — `Kv*dt/M` is ~3.9 against a limit of 2 — so a
commanded hold chattered at ~0.1 rad/s and sat 2–4 mm off its command.
`model_loader` selects `implicitfast`, which integrates that term: a 2 s hold
now reads 0.000 rad/s and <0.1 mm, and the open-loop grasp commit lands on its
commanded pose instead of ~2 mm off.

Simulation-only. On hardware the controller owns its own integration, so this
choice has no hardware equivalent. `tests/test_hardware.py --sim` pins it with
a sub-millimetre hold check.

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
    python main.py --hardware     # real robot: run_hardware.py

`main.py` is a minimal dispatcher: it chooses a front-end and nothing else.
Simulation is still the default so the project runs on a laptop; on the cell the
sensible default becomes the real robot and simulation should require `--sim`.
That is a one-line change in `main.py`'s `_parse`.

On macOS, use `mjpython` for the viewer:

    mjpython main.py

The front-ends can also be run directly:

    python run_simulation.py
    python run_hardware.py --ip 192.168.1.10

`run_hardware.py` imports no MuJoCo -- checked in `tests/test_ur_backend.py` by
importing the front-end in a fresh interpreter and asserting `mujoco` is absent
from `sys.modules`.

## Usage (simulation)

- `6` — spawn the cube on the table.
- `7` — **pick up / put down**. First press grasps, lifts and carries the cube
  to the home pose, where it waits in the jaws. The second press sets it back
  down. The cube is deliberately *not* released on the first press.
- `8` — pause / resume. The robot holds position and resumes where it left off.
- `9` — reset the scene.
- `.` — cycle the object's motion band: off / slow 15 / medium 30 / fast 45 mm/s.

Keys are `6`–`9` and `.` because the MuJoCo viewer binds every letter `a`–`z` to
one of its 26 render flags and digits `0`–`5` to geom groups, so nothing else is
free.

## Tests and harness

    python tests/test_layering.py            # robot/world separation
    python tests/test_dynamics.py            # analytic M(q)/g(q) vs MuJoCo
    python tests/test_ur_kinematics.py       # hardware chain FK/J/M/g vs MuJoCo
    python tests/test_ur_backend.py          # RTDE/Modbus plumbing, no robot
    python tests/test_task_flow.py           # pickup toggle: grasp -> hold -> put down
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
- `run_hardware.py` — **hardware front-end**: UR3e + 2F-85 + D435i, operator
  console, no MuJoCo.
- `run_tests.py` — headless trial harness; writes `summary.csv` and four graphs.
- `tests/test_layering.py` — enforces the robot / world separation.
- `tests/test_hardware.py` — validates hardware assumptions (sim + real modes).
- `tests/test_dynamics.py` — analytic `M(q)`/`g(q)` vs the simulator.
- `tests/test_ur_kinematics.py` — the hardware chain vs the simulator.
- `tests/test_ur_backend.py` — RTDE/Modbus plumbing against fakes.
- `models/ur3e/` — vendored UR3e MJCF and STL meshes.
- `models/ur3e/arm_dynamics.json` — the link data a hardware build loads for
  `M(q)`/`g(q)`; derived from the model once, shipped as data.
- `src/world.py` — **the boundary**: `TargetView` and the `ObjectWorld`
  interface.
- `src/world_sim.py` — the MuJoCo implementation of that boundary: scene,
  object pool, cube driver, camera pose.
- `src/task.py` — WAIT/TRACK/PICK/HOLD/PUTDOWN state machine. Pure policy: no
  mujoco, no viewer, no scene.
- `src/arm.py` — IK, trajectory execution, gripper, grasp readback.
- `src/calibration.py` — camera intrinsic calibration from rendered marker views.
- `src/camera.py` — RealSense D435i wrist camera: capture and deprojection.
- `src/config.py` — global flags and constants.
- `src/controller.py` — joint and LQR controllers, encoder model.
- `src/model_loader.py` — composes arm + gripper + wrist camera + table.
- `src/kinematics.py` — **the geometry seam**: the `Kinematics` interface, plus
  `AnalyticKinematics` (hardware-portable `M(q)`/`g(q)`).
- `src/kinematics_mujoco.py` — the MuJoCo implementation of that interface.
- `src/kinematics_ur.py` — the hardware implementation: chain FK/Jacobian from
  `arm_dynamics`, plus the gripper mount.
- `src/robot_backend.py` — **the I/O seam**: the `RobotBackend` interface.
- `src/robot_backend_mujoco.py` — the MuJoCo implementation: simulation clock,
  joint read/write, gripper readback, gravity feedforward.
- `src/robot_backend_ur.py` — the UR implementation: RTDE `servoj`, Modbus for
  the 2F-85, and the register map that turns the device into the grasp trio.
- `src/pick.py` — the robot-side pick sequence: search, approach, grasp, lift,
  carry. No MuJoCo, so the hardware front-end can use it as-is.
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
- Gravity compensation: a single feedforward term in the simulation backend,
  matching what a UR controller does internally. A commanded hold settles to
  **<0.1 mm** with no chatter (see *Gravity compensation* and *Integration*).
- Put-down: the pickup key toggles grasp → release → grasp, with the cube set
  back down gently at its own xy.
- The grasp is verified from the gripper's own registers, not assumed.
- All three test suites pass.

## Known limitation: grasp accuracy

**8 of 10** grasps succeed in a 10-trial headless run, with a mean gap of
**3.2 mm** between the gripper and the cube (the baseline was 5 of 10 at
5.7 mm).

The endpoint is no longer a contributor: with the integrator fixed, the
open-loop commit lands on its commanded pose to 0.01 mm, so the residual gap is
the vision estimate at commit time. The two remaining misses happen at 4-4.5 mm
of gap, well inside the jaws' ~12.5 mm clearance per side, so they are not a
lateral-offset failure. They come from contact on the way down:

- The descent interpolates the **joint** vector to the IK solution. A straight
  line in joint space is not a straight line for the tool: the pads swing up to
  33 mm along the path and cross the cube's top face ~7 mm off-centre (0.0 mm
  with a Cartesian path). Any contact there nudges the cube before the jaws
  close.
- The commit gate is loose in rotation: `SERVO_TRACK_TOL_ROT` is 0.15 rad
  (8.6°), so the jaws can be rotated relative to the cube and close on its
  corners.

The next step is a Cartesian commit descent, which removes the swing; tightening
the rotation gate is cheap and complementary.

A ~17 mm lateral offset on a 60 mm cube puts the pads against its side on the
way down; they deflect and close on air (`jaw_q` reads 0.785, fully closed, in
the failure).

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

## Done: the three providers

- **`src/robot_backend_ur.py`** — `URBackend`: RTDE `servoj` streaming for
  `step()`, `getActualQ`/`getActualQd` for state, and a Robotiq 2F-85 Modbus
  class for the registers `Arm.grasp_state` consumes. The clients are injected,
  so `tests/test_ur_backend.py` runs the whole thing against fakes.
- **`src/kinematics_ur.py`** — `URKinematics`: FK and the Jacobian from the
  6R chain in `arm_dynamics`, `M(q)`/`g(q)` from the same. The link data ships
  as `models/ur3e/arm_dynamics.json`, so no MuJoCo is needed on the robot.
  `tests/test_ur_kinematics.py` checks FK, `J`, `M` and `g` against MuJoCo to
  machine precision — offline.
- **`src/pick.py`** — the pick sequence moved out of `pick_place.py` so it no
  longer drags the simulator onto the robot. `run_hardware.py` uses it directly.
- **`run_hardware.py`** — composition root: builds the UR providers, the
  RealSense world, the task policy and an operator console. Imports no MuJoCo.

## Before the first run: measure these

Everything below is a placeholder in `src/config.py`, taken from the simulation.
They are the difference between "the code runs" and "the robot picks":

- **`UR_SERVO_KP` / `UR_SERVO_KV`** — the controller's stiffness, not published
  by UR. Identify from a step response; `tests/test_hardware.py --real`
  describes the procedure. The invariant to measure is the **time constant**
  `Kp/Kv`, not the damping ratio: `ζ` varies with pose through `M_jj`, whereas
  `Kp/Kv` is fixed by the servo.
- **`TOOL_MOUNT_POS` / `TOOL_MOUNT_QUAT`** — wrist_3_link to the gripper base
  (~10.8 mm and a 90° yaw in the model). A millimetre here is a millimetre on
  every grasp.
- **`CAM_MOUNT_POS` / `CAM_MOUNT_QUAT`** — gripper base to the D435i optical
  frame. A few millimetres biases every deprojected centre.
- **`TABLE_HEIGHT`** — as before; the sim's check proves the constant is
  load-bearing.

## Still unverified

- **`RealSenseSource`** has never been executed — no `pyrealsense2` wheel for
  Python 3.12 on macOS. The depth → deproject → detect path is exercised by
  `FakeDepthSource`; the device half is not.
- **`URBackend`** has never seen a controller. `tests/test_ur_backend.py` proves
  the plumbing and the register map; only `--real` proves the robot.
- **The 2F-85 force mapping** (`gCU` → N) is a monotone placeholder, chosen so
  the simulation's thresholds apply. Check the two endpoints on the unit.
- **Re-tune the integral and damping** when you enable them; the simulation
  numbers do not transfer.

Task logic (`src/task.py`), the servo law and its occlusion recovery, the EKF,
the LQR design and the grasp verification need no change.
