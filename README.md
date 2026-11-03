# Vision-Guided Pick: UR3e + RG2 + D435i

Name: Phillip Reiss  
Email: phillip.j.reiss@student.uts.edu.au  
Student ID: 26546453  
Subject: 42043 Robotics Studio 1

A UR3e arm with a wrist-mounted Intel RealSense D435i and an OnRobot RG2
gripper. The system detects a cube, tracks it, grasps it, holds it, and places
it at the pick location. The same robot code runs in MuJoCo and on the cell.

## Setup

```
python -m pip install -r requirements.txt
```

`pyrealsense2`, `ur_rtde`, `onRobot` and `pycurl` are hardware-only and are
documented in `requirements.txt`.

## Run

```
python main.py
python main.py --lightweight
python main.py --hardware --ip 192.168.0.194
python main.py --hardware --ip 192.168.0.194 --state-space
```

The hardware front-end defaults to the raw servo path. `--state-space` selects
the fitted observer/LQR path after a risk prompt; raw remains the default.
`--raw-servo` is the default alias.

Simulation keys: `6` spawn, `7` pick/place, `8` pause, `9` reset,
`.` object speed. Hardware keys: `t` tracking, `p` pick/place, `space`
pause, `h` home, `s` status, `v` snapshot when enabled, `q` quit.

The simulation control period matches the hardware: 8 ms, with four 2 ms
physics substeps. Arm moves, the observer and the target advance at 125 Hz.

## Method

The pipeline separates perception, estimation and control:

    depth frame
      -> object localisation -> TargetView
      -> target Kalman filter -> position and velocity
      -> PBVS -> joint reference
      -> state-space controller + arm observer -> joint command

- **Object localisation** deprojects the wrist depth image, removes the table,
  hand and out-of-bounds points, clusters the rest, fits the cube's top face and
  returns its centre and yaw. The hardware camera pose is forward kinematics
  composed with the measured flange-to-camera transform.
- **Target Kalman filter** is a six-state constant-velocity filter on the cube
  centre. It smooths intermittent depth measurements, bridges short gaps and
  supplies the velocity used for lead. It is used for moving targets; the static
  pick uses raw detections.
- **PBVS** is the outer loop. It builds the desired tool pose from the estimated
  cube position and yaw, computes the Cartesian pose error and converts it to a
  joint increment with damped least squares.
- **State-space controller** is the optional inner loop. It linearises the arm
  and joint servo about the current configuration, applies an LQR correction to
  the PBVS joint reference and sends the corrected command to the joint servo.
- **Arm observer** is a model-based estimator of joint position and velocity.
  It predicts through the arm dynamics and corrects position with the encoder;
  velocity is the unmeasured state supplied to the state-space controller.

The raw path sends the PBVS joint reference directly to the servo. The
state-space path inserts the observer/LQR correction. Both share the same
localisation, target filter, PBVS and task logic.

## Results

Simulation, four static random spawns, current sim camera mount:

| Trial | Outcome | Grasp gap |
|---|---|---|
| 0 | picked and lifted | 2.12 mm |
| 1 | picked and lifted | 1.91 mm |
| 2 | picked and lifted | 1.06 mm |
| 3 | picked and lifted | 1.07 mm |

Hardware, state-space session:

| Item | Result |
|---|---|
| Camera mount | recalibrated after a measured 92.5 degree mount rotation error |
| Table height | 12 mm |
| Servo preset | gain 1000, lookahead 0.05, 31 ms lag measured |
| Gripper close | 50 mm command, 59.7 and 59.9 mm measured at the cube |
| Tracking | position and rotation servoing while the cube moves |
| Picks | two, both held |
| Put-downs | two, both 0.0 mm from target and released |
| Videos | raw path and state-space path; links below |

State-space A/B on a 0.4 degree joint-1 step:

| Metric | Raw | State-space |
|---|---|---|
| final error | 0.001 deg | 0.004 deg |
| overshoot | 0.002 deg | 0.111 deg |
| settling | 0.216 s | 0.032 s |
| peak joint speed | 0.045 rad/s | 0.160 rad/s |

Observer fit: Kp 8000, Kv 400, qdot RMS 0.0078 rad/s. Prediction cost was
reduced from 10.79 ms to 2.41 ms by caching M(q) and g(q) to 0.3 degrees.
The state-space loop runs at about 45 Hz, not the 125 Hz servo rate; this is
visible as jerk at high pendant speed and is the main limit of the result.

KF on/off, four seeds per condition:

| Condition | KF | Success | Gap mean | Gap max |
|---|---|---:|---:|---:|
| static | on | 4/4 | 1.90 mm | 2.77 mm |
| static | off | 4/4 | 1.54 mm | 2.12 mm |
| slow moving | on | 4/4 | 2.62 mm | 3.28 mm |
| slow moving | off | 4/4 | 4.90 mm | 8.27 mm |

The filter helps a moving target, where it bridges intermittent looks and
supplies the lead velocity, but costs about 0.36 mm of static grasp gap.
The static pick therefore uses raw detections (use_kf=False); tracking uses
the filter. This is a deliberate trade-off, not a universal gain.

Fast band, twelve seeds per condition:

| Condition | KF | Success | Gap mean | Gap max |
|---|---|---:|---:|---:|
| fast moving | on | 12/12 | 2.53 mm | 6.07 mm |
| fast moving | off | 12/12 | 6.10 mm | 21.20 mm |

Natural-trial total across static, slow and fast bands: 28/28 successful, no
failed picks.

Failure counts:

| Set | Trials | Success | Failed |
|---|---:|---:|---:|
| Natural simulation (static, slow, fast) | 28 | 28 | 0 |
| Hardware picks (raw and state-space) | 4 | 4 | 0 |
| 65 mm/s sweep | 3 | 2 | 1 |
| 75 mm/s sweep | 3 | 2 | 1 |

The failed sweep trials are the 65/75 mm/s tracking losses described below,
not grasp or release failures.

Speed sweep, KF on, 8 s of motion:

| Speed | Trials | Success | Result |
|---:|---:|---:|---|
| 45 mm/s | 1 | 1 | no failure in the fast-band set above |
| 55 mm/s | 3 | 3 | clean tracking |
| 65 mm/s | 3 | 2 | first failure |
| 75 mm/s | 3 | 2 | failure repeats |

The first failure is at **65 mm/s**. The failed trial gave an estimate error of
192 mm and the approach did not complete, so no grasp was attempted. Above
75 mm/s the single-seed sweep generally failed; one 115 mm/s seed succeeded,
so the boundary is stochastic rather than a sharp cutoff.

Evidence logs: [static KF](run_logs/run_tests_2026-11-03_161056.txt),
[slow KF](run_logs/run_tests_2026-11-03_161133.txt),
[fast KF 1](run_logs/run_tests_2026-11-03_162122.txt),
[fast KF 2](run_logs/run_tests_2026-11-03_164015.txt),
[45 mm/s](run_logs/run_tests_2026-11-03_164400.txt),
[55 mm/s](run_logs/run_tests_2026-11-03_164607.txt),
[65 mm/s](run_logs/run_tests_2026-11-03_164622.txt),
[75 mm/s](run_logs/run_tests_2026-11-03_164639.txt).

Offline verification:

    preflight_cell.py        23/23
    test_hardware.py --sim   30/30
    test_ur_kinematics       pass
    test_gripper_backend     pass
    test_task_flow           pass

Limits: two hardware sessions and four picked/placed cubes in total (two raw,
two state-space), not a repeated-trial study. The simulation uses a separate
camera mount because the cell bracket was moved; the hardware and simulation
are not identical geometry.

## Roadblocks

Over the course of the project, I encountered a number of roadblocks that had to be overcome. 

| # | Roadblock | What happened | Consequence | What was done |
|---|---|---|---|---|
| 1 | Hardware platform constraint | macOS could not reliably handle the D435i USB connection; work moved to an older ThinkPad running Debian | Lower compute headroom; state-space loop limited to about 45 Hz; video shot at 80% pendant speed | Accepted the limit; 45 Hz and 80% are reported as the main state-space constraints |
| 2 | Library/toolchain mismatch | The older laptop and the required onRobot client did not line up cleanly; onRobot also depends on pycurl, which it does not declare | Setup time lost to dependency and compatibility work before the cell could run | Pinned the working dependency set; preflight checks ur_rtde, pycurl, onRobot, RealSense and OpenCV |
| 3 | Calibration approach abandoned under time pressure | ArUco and ruler-based camera calibration were attempted and abandoned; the fallback was cruder than intended | The fallback did not detect that the physical camera mount was in the wrong orientation | Recalibrated the mount in the next session at the measured 92.5 degree correction |
| 4 | Near-wasted session and sim/hardware divergence | The wrong mount pose invalidated the 11/02 session's pose and deprojection; simulation still used the old mount | One session largely lost; simulation and hardware camera geometry diverged; simulation initially fell to 0/3 | Recalibrated hardware; added separate SIM_CAM values; simulation restored to 4/4 |

The causal chain is the important part:

    macOS USB failure
      -> older laptop
      -> library mismatch and lost setup time
      -> calibration shortcut under time pressure
      -> wrong camera mount pose
      -> near-wasted session
      -> hardware/sim camera divergence

## Videos

| Video | Log | Content |
|---|---|---|
| [42043_integrated_project_raw.mp4](42043_integrated_project_raw.mp4) | [run_2026-11-02_140524.txt](run_logs/run_2026-11-02_140524.txt) | Raw servo path: two picks at 60.0 and 60.4 mm jaw width, two put-downs at 0.0 mm from target. |
| [42043_integrated_project_state_space.mp4](42043_integrated_project_state_space.mp4) | [run_2026-11-03_112801.txt](run_logs/run_2026-11-03_112801.txt) | State-space path at 80% pendant speed: two picks at 59.7 and 59.9 mm jaw width, two put-downs at 0.0 mm from target. |

## Calibration

Camera mount and pick height are measured values in `src/config.py`. The
consolidated calibration tool is:

```
python tools/calibrate_mount.py
```

Speed calibration:

```
python tools/calibrate_speed.py --stage low
python tools/calibrate_speed.py --stage full
```

The state-space observer is fitted and A/B-tested; the current config carries
Kp 8000 and Kv 400. The fitting workflow is:

```
python tools/validate_state_space.py record --joint 1 --step 1.5
python tools/validate_state_space.py fit --record <record>.csv --write
python tools/validate_state_space.py ab --fit <fit>.json --joint 1 --step 1.5
```

## Tests

```
python tests/preflight_cell.py
python tests/test_arm_control.py --config-only
python tests/test_hardware.py --sim
python tests/test_gripper_backend.py
python tests/test_yaw_sweep.py
python run_tests.py --trials 4
python run_tests.py --trials 3 --kf on --speed 65 --move-s 8
```

The last command runs the 65 mm/s tracking sweep point; `--speed` is in mm/s
and overrides the named band.

`preflight_cell.py` is the offline gate. Hardware acceptance requires
`test_arm_control.py` and `test_camera_enum.py` on the cell.

Run logs are written to `run_logs/` with the command arguments, working
directory, Python version and timestamp at the top.

## Architecture

Three interfaces are shared, with simulation and hardware implementations:

| Interface | Simulation | Hardware |
|---|---|---|
| `src/kinematics.py` | `kinematics_mujoco.py` | `kinematics_ur.py` |
| `src/robot_backend.py` | `robot_backend_mujoco.py` | `robot_backend_ur.py` |
| `src/world.py` | `world_sim.py` | `world_real.py` |

`src/arm.py`, `src/servo.py`, `src/pick.py` and `src/task.py` contain no
MuJoCo or device driver. The hardware world is wrist-mounted only.

## References

| Source | Used for | Licence |
|---|---|---|
| [UTS-RI/UR3_RG2_rosbridge](https://github.com/UTS-RI/UR3_RG2_rosbridge) | UR3e MJCF, meshes, RG2 XML-RPC method | MIT, © 2026 Robotics Institute, UTS |
| [tonydle/OnRobot_ROS2_Description](https://github.com/tonydle/OnRobot_ROS2_Description) | RG2 geometry and meshes | MIT, © 2025 Tony Le |
| [Osaka-University-Harada-Laboratory/onrobot](https://github.com/Osaka-University-Harada-Laboratory/onrobot) | upstream RG2 meshes | MIT |

Model changes and rationale are recorded in `models/README.md`. Cell
measurements are recorded at their point of use in `src/config.py`.

## Statement on the Use of Generative AI

During the course of this assignment I used the DeepSeek model to assist me in all areas of the project. The primary areas I used it for include:

- Generating boilerplate and application code
- Implementing algorithms and control architecture
- Generating test harnesses
- Designing experiments and analysing results
- Diagnosing hardware and software failures
- Annotating code and reviewing/refactoring it
- Editing and formatting technical documentation
- Managing, condensing or removing orphaned files
- Auditing code and requirements for completeness, correctness and bugs
