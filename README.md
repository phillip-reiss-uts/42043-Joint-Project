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

**Two things the seam does not solve.** `mass_matrix`/`gravity_torque` have no
clean UR3e equivalent — the controller will not hand you an inertia matrix, so a
real provider needs a pinned or system-identified model. And `servo_gains()`
returns the *simulated* actuator's stiffness, which is what `A_cl` is built
from; a real one needs the controller's actual servo dynamics, which UR does not
publish precisely. Both are flagged in the source.

`arm.move` still steps the simulation directly (`mj_step`), because executing a
joint trajectory needs a clock. On hardware that call becomes a streamed
`servoj`/`movej`; the pacing, easing, pause and abort logic around it does not
change.

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

The seam is in place, but the port itself is not done. What remains is a second
`ObjectWorld` implementation plus real drivers:

- a `RealObjectWorld` backed by RealSense captures, replacing `perceive()`;
- forward kinematics from the UR3e controller for `gripper_pose()`;
- UR RTDE/URScript for joint moves. `Arm.move` currently integrates a joint
  trajectory against MuJoCo's position servos; on hardware this becomes a
  streamed move command. The pacing, easing, pause and abort logic around it
  does not change;
- Modbus reads for the 2F-85 registers that `Arm.grasp_state` already consumes.

Task logic (`src/task.py`), the servo law and its occlusion recovery, the EKF,
the LQR design and the grasp verification need no change. Replacing the
kinematics calls in `arm.py`/`servo.py`/`observer.py` (Jacobians, mass matrix)
with the real controller's FK is the remaining work inside the robot layer.
