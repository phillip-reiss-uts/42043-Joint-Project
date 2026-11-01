# Vision-Guided Pick (UR3e + RG2 + D435i)

A UR3e with a wrist-mounted RealSense D435i and an OnRobot RG2 that tracks a
cube, grasps it on command, and holds it until told to put it down. Runs in
MuJoCo and on the real robot from the same robot code.

## Setup

    python -m pip install -r requirements.txt

`pyrealsense2` is hardware-only and is not in `requirements.txt`.

## Run

    python main.py                # simulation (default)
    python main.py --lightweight  # simulation, no camera panels (older machines)
    python main.py --hardware --ip 192.168.0.194
    python main.py --hardware --ip 192.168.0.194 --raw-servo  # original direct-servo path

Simulation keys: `6` Spawns a cube in a random position and orientation,
`7` pick up / put down (toggle), `8` pause, `9` reset,
`.` Cycles object speed, from 0 to 45 mm /s. On macOS use `mjpython main.py`.

Hardware keys: `p` pick up / put down (toggle), `t` start/stop tracking,
`space` pause / resume, `h` go to home, `s` status, `q` quit. Both front-ends
print their keys at startup.

Hardware defaults to the state-space joint path (`ArmObserver` + LQR around the
PBVS joint reference). `--raw-servo` restores the original direct command path;
use it if the observer has to be removed at the cell.

The two sets differ because the scene keys have nothing to act on: you place the
cube by hand instead of spawning it, and there is no scene reset. After an
abandoned pick, `h` with the jaws open is the recovery — it re-homes and releases
the carry.

`space` pauses the hardware run: it holds the arm at its last commanded position,
stops the mode advancing and ignores `p`, so you can reach into the cell without
the robot starting a pick under your hand. It does **not** abort a pick already in
progress — `task.step()` blocks for the whole pick and the key is only read
between them, so the pendant remains the way out of one.

On hardware, perception prints one `[PERC]` line per look — points that survived
the height filter, cluster count, and the chosen centre, span and point count.
That is the only view of what the camera sees; `PERCEPTION_LOG_S` in
`src/config.py` sets the interval, and `0` disables it.

`--lightweight` drops the RGB-image and point-cloud panels. Each is its own GL
render on top of the depth pass perception already does, so on a machine without
a decent GPU they dominate the frame and a pick takes minutes of wall time
instead of seconds. The 3D scene, the tool trail, the detections and the text
panels all stay — the robot uses none of the panels.

### Snapshots on hardware (opt-in)

    python main.py --hardware --ip 192.168.0.194 --snapshot shots/

arms a `v` key that writes `shots/snap_001.png` — an annotated picture of the
**last camera look**: false-colour depth, the detected hull, the object centre
with its world coordinates, and the current tool position. It is the only way to
see *why* a look failed (a hand in the cloud, jaws clipping the cube, a depth
dropout), because the hardware run has no viewer.

Off unless `--snapshot` is given, and off means off: no frame is cached, no
behaviour changes, and `src/liveview.py` is not imported. It never re-captures —
it draws the frame perception already took, so it cannot consume a frame or delay
a command. There is deliberately no live window: this runs over SSH, and a draw
inside the servo loop would starve the 8 ms servoj stream.

## State-space validation

Hardware defaults to the state-space joint path. It is not valid until the
observer model has been fitted and the controller compared with the raw path:

    python tools/validate_state_space.py record --joint 1 --step 1.5
    python tools/validate_state_space.py fit --record state_space_record_<stamp>.csv --write
    python tools/validate_state_space.py ab --fit state_space_record_<stamp>_fit.json --joint 1 --step 1.5

`record` logs the raw joint step and observer estimate. `fit` replays it
offline for `UR_SERVO_KP/KV` and refuses a qdot residual above 0.05 rad/s.
`ab` is required before using state-space for a pick; it refuses a failed fit
and compares final error, overshoot, settling and peak joint speed.

## Speed calibration

Hardware servoing speed is characterised before a fast run:

    python tools/calibrate_speed.py --stage low     # pendant 10%, safety checks
    python tools/calibrate_speed.py --stage full    # pendant 100%, find the scale

The tool has an independent hard command limit, joint-speed limit, path box,
following-error limit and timeout. It stops at the first failure and never
tries a faster candidate. It writes `UR_SPEED_SCALE` only with `--write`.

## Camera calibration

`CAM_POS`/`CAM_QUAT` in `src/config.py` are the current measured mount. Every
detected centre inherits any error directly, so recalibrate on a new cell or
after moving the bracket.

`tools/calibrate_mount.py` consolidates the measurements: table height, camera
height, translation, angle, pick height and grip width, plus the multi-position
translation and wrist-jog checks. The manual offset loop is the fallback the
tool is built around:

1. Jog the tool until it is centred on the cube. That TCP xy is the truth in
   the base frame.
2. Take one perception look and record the detected cube xy.
3. Jog the tool to the detected xy. The physical gap between the tool and the
   cube centre is the error in `CAM_POS`.
4. Add that offset to `CAM_POS`, restart, and repeat from 2 until the arm
   centres on the cube to within ±2 mm.

`CAM_POS_z` is measured, not set by the loop: on this cell the camera sits
114 mm below tool0 (tool0 300 mm, camera 186 mm), so
`CAM_POS_z = 114 mm — T`, where `T` is the flange-to-gripper-top distance.

The loop constrains translation only. `CAM_QUAT` is set by the mount geometry,
not by this loop, and is good enough for this task.

The marker tooling is still in the tree, but the wrist ArUco route was tried on
2026-10-30 and did not converge — the ruler check read 65.9 mm error at truth
(0, -0.25). It is not the calibration route for this cell's wrist today; the
offset loop above is. The tool:

    python tools/calibrate_camera.py --aruco     # printed marker, several poses
    python tools/calibrate_camera.py --touch     # touched cube positions

Both produce `CAM_POS` / `CAM_QUAT` only. A camera on a stand is not covered:
the fixed-camera mode, its config (`CAMERA_MODE`, `FIXED_CAM_*`) and its
calibration routines have been removed, and the camera rides the gripper.

## Tests

    python tests/test_layering.py        # robot/world + provider separation
    python tests/test_task_flow.py       # pickup toggle: grasp -> hold -> put down
    python tests/test_dynamics.py        # analytic M(q)/g(q) vs MuJoCo
    python tests/test_ur_kinematics.py   # hardware chain vs MuJoCo
    python tests/test_kinematics_parity.py  # TCP pose + frame convention
    python tests/test_ur_backend.py      # RTDE/RG2 plumbing, no robot
    python tests/test_calibration_camera.py  # extrinsic solvers, synthetic
    python tests/test_liveview.py        # snapshot projection/drawing, and OFF
    python tests/test_hardware.py --sim  # hardware assumptions
    python tests/test_gripper_backend.py # RG2 XML-RPC, fakes only
    python tests/preflight_cell.py       # the gate: everything offline, before travel
    python run_tests.py --trials 6       # end-to-end picks, writes test_output/

## Architecture

Three interfaces are shared; each has a simulation and a hardware implementation,
chosen by the entry point and injected into the robot layer.

| interface | simulation | hardware |
|---|---|---|
| `src/kinematics.py` | `kinematics_mujoco.py` | `kinematics_ur.py` |
| `src/robot_backend.py` | `robot_backend_mujoco.py` | `robot_backend_ur.py` |
| `src/world.py` | `world_sim.py` | `world_real.py` |

`src/pick.py`, `src/servo.py`, `src/arm.py` and `src/task.py` contain no MuJoCo
and no device driver. `tests/test_layering.py` enforces it.

The hardware world is the wrist camera only: `RealObjectWorld` in
`src/world_real.py` builds the camera pose as `FK(q) @ T_flange_cam`. The
eye-to-hand variant (`CAMERA_MODE = 'fixed'`, `FixedCameraWorld`) has been
removed; a camera on a stand is outside what the hardware path supports.

## References

This project uses other people's descriptions and follows their methods.


| source | used for | licence |
|---|---|---|
| [UTS-RI/UR3_RG2_rosbridge](https://github.com/UTS-RI/UR3_RG2_rosbridge) | the UR3e MJCF and its seven STL meshes (`models/ur3e/`), and the OnRobot XML-RPC method the RG2 backend is built on (`examples/15_real_gripper_control.py`) | MIT, © 2026 Robotics Institute, UTS |
| [tonydle/OnRobot_ROS2_Description](https://github.com/tonydle/OnRobot_ROS2_Description) | the RG2 geometry: four-bar linkage dimensions and meshes (`models/onrobot_rg2/`) | MIT, © 2025 Tony Le |
| [Osaka-University-Harada-Laboratory/onrobot](https://github.com/Osaka-University-Harada-Laboratory/onrobot) | upstream origin of the RG2 meshes, per the package above | MIT |

The UR3e meshes are byte-for-byte copies. What was changed in
each description, and why, is recorded in [models/README.md](models/README.md);
the licence texts sit beside the files they cover.

## Documentation

This README is authoritative. Design rationale and the measurements behind these
constants live in comments at the point of use -- `src/config.py` in particular,
where each cell constant records what it was measured from.
