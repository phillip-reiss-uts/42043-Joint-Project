# Vision-Guided Pick: UR3e + RG2 + D435i

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

## Results

Simulation, four static random spawns, current sim camera mount:

| Trial | Outcome | Grasp gap |
|---|---|---|
| 0 | picked and lifted | 2.12 mm |
| 1 | picked and lifted | 1.91 mm |
| 2 | picked and lifted | 1.06 mm |
| 3 | picked and lifted | 1.07 mm |

Hardware, one state-space session:

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
```

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
