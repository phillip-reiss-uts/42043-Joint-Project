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
the experimental observer/LQR path after a risk prompt. `--raw-servo` is the
default alias.

Simulation keys: `6` spawn, `7` pick/place, `8` pause, `9` reset,
`.` object speed. Hardware keys: `t` tracking, `p` pick/place, `space`
pause, `h` home, `s` status, `v` snapshot when enabled, `q` quit.

The simulation control period matches the hardware: 8 ms, with four 2 ms
physics substeps. Arm moves, the observer and the target advance at 125 Hz.

## Results

Simulation, four static random spawns:

| Trial | Outcome | Grasp gap |
|---|---|---|
| 0 | picked and lifted | 2.33 mm |
| 1 | picked and lifted | 0.70 mm |
| 2 | picked and lifted | 3.13 mm |
| 3 | picked and lifted | 1.91 mm |

Hardware, one cell run:

| Item | Result |
|---|---|
| Camera mount | recalibrated after a measured 92.5 degree mount rotation error |
| Table height | 12 mm |
| Gripper close | 50 mm command, 60.5 mm measured at the cube |
| Tracking | the arm servos while the cube is moved and rotated |
| Pick | grasped, held and carried after the target settles |
| Place | released at the recorded pick xy; 0.0 mm from the target |
| Video | 42043_integrated_project_video.mp4, including the tracking phase |

Offline verification:

    preflight_cell.py        23/23
    test_hardware.py --sim   30/30
    test_ur_kinematics       pass
    test_gripper_backend     pass
    test_task_flow           pass

State-space is an experimental path. A one-joint step fit gave Kp 125,
Kv 400 and qdot RMS 0.0091 rad/s. The observer runs at about 45 Hz because
M(q) and g(q) are evaluated every cycle, and the measured hardware response
has a 1.3 s lag that the current model does not represent. The fitted gains
are therefore not part of the delivered pick.

Limits of the evidence: the hardware result is one successful run that
includes moving/rotating-target tracking, but it is not a repeated-trial
study. The state-space path is not used in the video.

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

The state-space observer is optional and must be fitted and A/B-tested:

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
