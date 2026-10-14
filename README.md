# Vision-Guided Pick (UR5e + 2F-85 + D435i)

MuJoCo simulation for a vision-guided pick task on a UR5e arm with a
wrist-mounted RealSense D435i RGB-D camera and a Robotiq 2F-85 gripper.

## Setup

    pip install -r requirements.txt

## Run

    python main.py

## Usage

 - `6` — spawn the cube on the table (respects the hard-mode spawn flags).
 - `7` — start / stop the search.
 - `8` — pause / resume. The robot holds position and resumes where it left off.
 - `9` — reset the scene.
 - `.` — cycle the object's motion band: off / slow 15 / medium 30 / fast 45 mm/s.

Keys are 6-9 and `.` because the MuJoCo viewer binds every letter a-z to one of
its 26 render flags and the digits 0-5 to geom groups, so nothing else is free.

## Layout

- `main.py` — entry point: state machine, command keys, scene setup.
- `run_tests.py` — headless trial harness; writes `summary.csv` and four graphs.
- `src/arm.py` — IK, trajectory execution, gripper, and the grasp readback.
- `src/calibration.py` — camera intrinsic calibration from rendered marker views.
- `src/camera.py` — RealSense D435i wrist camera: capture and deprojection.
- `src/config.py` — global flags and constants.
- `src/controller.py` — state-space joint controller and encoder.
- `src/model_loader.py` — composes UR5e + 2F-85 + wrist camera + table.
- `src/observer.py` — observer for the unmeasured joint states.
- `src/overlay.py` — live viewer overlays (trail, frustum, detections, camera view).
- `src/perception.py` — clustering, convex hull, calipers, object detection.
- `src/pick_place.py` — scene, spawn/despawn, one pick, table clearing.
- `src/servo.py` — PBVS servo, occlusion recovery, open-loop grasp commit.
- `src/state_space.py` — linearisation, controllability/observability, LQR design.
- `src/target_motion.py` — moving-target driver and the speed bands.
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