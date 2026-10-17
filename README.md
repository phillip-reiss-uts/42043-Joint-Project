# Vision-Guided Pick (UR3e target + 2F-85 + D435i)

MuJoCo simulation for a vision-guided pick task on a UR-series arm with a
wrist-mounted RealSense D435i RGB-D camera and a Robotiq 2F-85 gripper.

## Target hardware

The intended real platform is a **UR3e + RealSense D435i + Robotiq 2F-85**.
The simulation currently loads a UR5e because the installed `mujoco_menagerie`
does not ship a UR3e; see `ROBOT_ARM` in `src/config.py`. Nothing in the robot
code depends on which arm is loaded, so switching is a one-line config change
once a UR3e MJCF is available.

## Architecture: the robot / world boundary

Robot code must run on real hardware. The object side is a simulation
demonstration. Those two concerns are separated by `src/world.py`:

    robot side (hardware-portable)      world side (simulation)
    ------------------------------      ----------------------
    task.py       (WAIT/SEARCH FSM)     world.py       (boundary interface)
    arm.py        (joints, gripper)     pick_place.py  (scene, spawn, pool)
    servo.py      (PBVS)                target_motion.py (cube glide, sim only)
    controller.py (PD / LQR)            camera.py      (RGB-D sensor)
    observer.py   (EKF)                 model_loader.py (scene composition)
    state_space.py (linearisation)      overlay.py     (viewer overlays)
    tracking.py   (Kalman filter)       run_simulation.py (sim front-end)

The robot reaches the world **only** through `ObjectWorld`, which exposes two
read-only methods: `perceive()` returning `TargetView` records (camera-derived
estimates — never ground truth), and `gripper_pose()` (the robot's own pose).
There is deliberately no method for moving or spawning an object: the robot has
no business commanding the object, and that capability is exactly how the two
sides became coupled before.

`tests/test_layering.py` enforces the rule mechanically. Run it after any
change:

    python tests/test_layering.py

It fails if a robot module imports `target_motion`, `pick_place`,
`model_loader`, `camera`, `calibration` or `overlay`, or touches object state.

Note `task.py` is the strictest case: it contains no `mujoco` at all, so the
task policy — when to search, what a pause means mid-motion, what an
interrupted search implies — is expressed independently of any simulator.

## The kinematics seam

The second boundary is `src/kinematics.py`. FK, the Jacobian, `M(q)`, `g(q)` and
the servo gains all come from a `Kinematics` provider rather than from direct
MuJoCo calls in the control layer:

    robot layer                          provider
    -----------                          --------
    arm.solve_ik        pose_and_jacobian
    servo.joint_step    jacobian          MujocoKinematics  (today)
    observer            mass_gravity, servo_gains
    state_space         mass_matrix, gravity_torque, servo_gains
                                          URKinematics      (on hardware)

`MujocoKinematics` wraps the exact calls the project already made, so wiring it
in is numerically neutral — verified bit-identical for IK, `M(q)`, `g(q)`,
`A_cl`/`B_cl`, and the observer's `K_ss`/`A_d`/`Q_d`.

The layering test now fails if any robot module calls `mj_kinematics`,
`mj_jacBody`, `mj_fullM`, `mj_crb`, `mj_rne`, `mj_factorM` or `mj_comPos`
directly. `mj_step` is exempt: it is the physics-clock question, not a
kinematics query.

### M(q), g(q) and the servo gains

A UR3e controller does not hand you an inertia matrix or its servo stiffness, so
both are **modelled** rather than queried.

`src/arm_dynamics.py` computes `M(q)` and `g(q)` from link data with a composite
rigid-body formulation (no simulator). `AnalyticKinematics` wraps it as a
provider. It is **verified to machine precision** against MuJoCo — worst
`|dM| = 2.9e-15`, worst `|dg| = 3.2e-14` over nine configurations, `M`
symmetric and positive definite. Run that check with:

    python tests/test_dynamics.py

Three things had to be handled that the URDF does not make obvious:

- **Rotor inertia.** `dof_armature = 0.1 kg m²` per joint on this model, against
  wrist link inertias of ~0.0025 — a 40× error if omitted. It is a *required*
  constructor argument precisely so a caller cannot forget it.
- **The gripper's internal joints.** The 2F-85 has finger joints of its own, so
  the distal subtree cannot be merged by "stop at the first jointed body". The
  whole subtree is folded into `wrist_3` as one rigid body (1.3135 kg).
- **The `base` body's 180° rotation**, which every downstream frame inherits.

`servo_gains()` is still the open question: `A_cl` is built from `Kp`/`Kv`, and
those describe the *simulated* position servo (2000/400 on the first three
joints, 500/100 on the wrist). `AnalyticKinematics` requires the gains to be
passed explicitly rather than defaulted, so a hardware build has to supply the
controller's real stiffness — identify it from a step response, since UR does
not publish it.

## The robot backend seam

The third boundary is `src/robot_backend.py`: **who owns the clock**, and how
joint state is read and written. `arm.py` and `servo.py` used to do this inline:

    data.ctrl[:ARM_DOF] = q_cmd
    data.ctrl[model.nu - 1] = GRIP_OPEN
    mujoco.mj_step(model, data)

Those four lines encode three hardware assumptions — arm commands are a joint
vector at one address range, the gripper is the last actuator, and a control
cycle is `mj_step`. None hold on a UR3e. `RobotBackend` exposes:

    joint_positions() / joint_velocities() / time() / dt
    command(q_arm, grip)      # set targets WITHOUT advancing time
    step()                    # advance one control cycle
    gripper_state()           # {'q', 'v', 'force'} -- the Modbus trio
    joint_addresses(name)     # model handles

With this in place **`arm.py` and `servo.py` contain no `import mujoco` at all.**
Geometry comes from `kinematics.py`, the clock from `robot_backend.py`. Verified
bit-identical over a 400-step trajectory (`qpos`, `qvel`, `ctrl`, `time`), and
the servo regression is unchanged (41 of 44 recorded fields bit-identical, the
rest at 1e-8).

`MujocoBackend` is the simulation implementation. A `URBackend` would map
`command`/`step` onto RTDE `servoj` (where `step` blocks for the controller's
cycle) and `gripper_state` onto 2F-85 Modbus registers.

**The residual gap is timing, not logic.** A real `step()` blocks for the
control period and `read()` returns noisy encoder values, whereas the simulated
backend steps instantly and reads perfect state. That difference is exactly what
`ArmObserver` exists to absorb — which is why it takes encoder noise and torque
disturbance seriously rather than treating them as decoration.

## Setup

    pip install -r requirements.txt

## Run

    python main.py                # simulation (current default)
    python main.py --sim          # simulation, explicit
    python main.py --hardware     # real robot (not implemented yet)

`main.py` is a minimal dispatcher: it chooses a front-end and nothing else.
With no option it runs the simulation. Once the hardware front-end exists the
sensible default becomes the real robot and simulation should require `--sim`;
that is a one-line change in `main.py`'s `_parse`.

The simulation front-end can also be run directly:

    python run_simulation.py

Demonstrations (each needs to compose the scene, so they live outside the
robot modules):

    python -m src.demo_servo        # PBVS vs open-loop planning
    python -m src.demo_controller   # PD vs LQR outer loop
    python -m src.demo_observer     # velocity estimation from encoders
    python -m src.demo_state_space  # controllability / observability

## Usage

 - `6` — spawn the cube on the table (respects the hard-mode spawn flags).
 - `7` — start / stop the search.
 - `8` — pause / resume. The robot holds position and resumes where it left off.
 - `9` — reset the scene.
 - `.` — cycle the object's motion band: off / slow 15 / medium 30 / fast 45 mm/s.

Keys are 6-9 and `.` because the MuJoCo viewer binds every letter a-z to one of
its 26 render flags and the digits 0-5 to geom groups, so nothing else is free.

## Layout

- `main.py` — minimal entry point: dispatches to a front-end (`--sim` default,
  `--hardware` reserved).
- `run_simulation.py` — **simulation front-end**: passive viewer, keyboard,
  scene reset, the moving cube. Imports mujoco freely; nothing here is for
  hardware.
- `run_tests.py` — headless trial harness; writes `summary.csv` and four graphs.
- `tests/test_layering.py` — enforces the robot / world separation.
- `tests/test_hardware.py` — validates hardware assumptions (sim + real modes).
- `tests/test_dynamics.py` — analytic `M(q)`/`g(q)` vs the simulator.
- `src/world.py` — **the boundary**: `TargetView`, `ObjectWorld`,
  `SimObjectWorld`.
- `src/task.py` — WAIT/SEARCH state machine, pause and abort semantics. Pure
  policy: no mujoco, no viewer, no scene.
- `src/arm.py` — IK, trajectory execution, gripper, grasp readback. Knows
  nothing about objects or the viewer.
- `src/calibration.py` — camera intrinsic calibration from rendered marker views.
- `src/camera.py` — RealSense D435i wrist camera: capture and deprojection.
- `src/config.py` — global flags and constants.
- `src/controller.py` — joint and LQR controllers, encoder model.
- `src/model_loader.py` — composes arm + gripper + wrist camera + table.
- `src/kinematics.py` — **the geometry seam**: FK/Jacobian/M/g providers.
- `src/robot_backend.py` — **the I/O seam**: stepping, joint read/write, gripper.
- `src/arm_dynamics.py` — Newton-Euler `M(q)`/`g(q)` from link data.
- `src/world_real.py` — **hardware world**: RealSense depth behind `ObjectWorld`.
- `src/observer.py` — observer for the unmeasured joint states.
- `src/overlay.py` — live viewer overlays (trail, frustum, detections, camera view).
- `src/perception.py` — clustering, convex hull, calipers, object detection.
- `src/pick_place.py` — scene, spawn/despawn, one pick, table clearing.
- `src/servo.py` — PBVS servo, occlusion recovery, open-loop grasp commit.
- `src/state_space.py` — linearisation, controllability/observability, LQR design.
- `src/target_motion.py` — moving-target driver and the speed bands (sim only).
- `src/tracking.py` — constant-velocity Kalman filter on the vision estimate.
- `requirements.txt` — dependencies.

## Status

- The robot picks objects from the table and places them in a chute.
- The table surface is filtered by Z height, and the robot searches for objects
  that are partly hidden. A fully hidden object is still not found.
- The object can be commanded to move while the robot is servoing, to show the
  tracking holding up. The robot keeps servoing while it moves, falls back to
  the search routine if it leaves the view entirely, and only commits to the
  grasp once it has watched the object hold still for 3 s.
- The grasp is verified from the gripper's own position and force rather than
  assumed from the servo completing, with a lift-and-recheck when it reads empty.
- Move speeds and gripper strength are configurable.
- User controls are responsive and reliable.
- Behaviour is deterministic and repeatable.

## Porting to hardware

### Done: `src/world_real.py`

`RealObjectWorld` implements the boundary against a RealSense depth stream.
`pyrealsense2` is imported lazily, so the module imports cleanly on a machine
without librealsense, and `FakeDepthSource` exercises the whole
depth → deproject → detect path in CI. The three things the simulator gave for
free are now explicit:

- **extrinsics** — camera pose is `FK(q) @ T_flange_cam`, so the mount offset
  must be measured. A few millimetres is a direct bias on every deprojected
  centre, the same failure the servo's occlusion notes describe.
- **intrinsics** — read from the device, so `fx != fy` and an off-centre
  principal point are handled rather than assumed away. `check_intrinsics()`
  reports the difference instead of asserting equality.
- **depth conventions** — `uint16` millimetres scaled to metres, converted
  explicitly; MuJoCo's metres/z-forward convention is not assumed.

`TABLE_HEIGHT` is still a config constant, as agreed — the harness now has a
check proving it is load-bearing (with a cube on the table, a plane 20 mm too
high swallows most of it), and a `--real` check telling you to measure the true
surface.

### Still to do

- **`URBackend`** — RTDE `servoj`/`movej` for `step()`, Modbus for the 2F-85
  registers `Arm.grasp_state` already consumes. `Arm.move` currently integrates a
  joint trajectory against MuJoCo's position servos; on hardware this becomes a
  streamed move command. The pacing, easing, pause and abort logic around it does
  not change.
- **`URKinematics`** — FK/Jacobian from the controller, with `arm_dynamics`
  supplying `M(q)`/`g(q)`.
- **Camera mount + table plane** — measure `T_flange_cam` and `TABLE_HEIGHT`.
- **Rotor inertia and `Kp/Kv`** — identify both (`tests/test_hardware.py --real`
  describes the procedure for each).

Task logic (`src/task.py`), the servo law and its occlusion recovery, the EKF,
the LQR design and the grasp verification need no change.
