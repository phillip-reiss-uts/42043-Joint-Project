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

Simulation keys: `6` Spawns a cube in a random position and orientation,
`7` pick up / put down (toggle), `8` pause, `9` reset,
`.` Cycles object speed, from 0 to 45 mm /s. On macOS use `mjpython main.py`.

Hardware keys: `p` pick up / put down (toggle), `space` pause / resume, `h` go to
home, `s` status, `q` quit. Both front-ends print their keys at startup.

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

## Camera calibration

`CAM_POS`/`CAM_QUAT` in `src/config.py` are the simulator's bracket, not a
measurement. Every detected centre inherits that error directly, so calibrate
before a hardware run.

The route that works on this cell is a direct offset loop against the tool:

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

The marker tooling is still in the tree for the fixed-camera fallback below, but
the wrist ArUco route was tried on 2026-10-30 and did not converge — the
ruler check read 65.9 mm error at truth (0, -0.25). It is not the calibration
route for this cell's wrist.

A camera on a stand instead of the wrist (`CAMERA_MODE = 'fixed'`) is the
fallback. There is no fixed-camera routine in this tool any more: a camera on a
stand is aimed by hand and measured with a tape, and the wrist mount is what
`--aruco` and `--touch` calibrate:

    python tools/calibrate_camera.py --aruco     # printed marker, several poses
    python tools/calibrate_camera.py --touch     # touched cube positions

## Tests

    python tests/test_layering.py        # robot/world + provider separation
    python tests/test_task_flow.py       # pickup toggle: grasp -> hold -> put down
    python tests/test_dynamics.py        # analytic M(q)/g(q) vs MuJoCo
    python tests/test_ur_kinematics.py   # hardware chain vs MuJoCo
    python tests/test_kinematics_parity.py  # TCP pose + frame convention
    python tests/test_ur_backend.py      # RTDE/RG2 plumbing, no robot
    python tests/test_fixed_camera.py    # eye-to-hand world, offline (PENDING)
    python tests/test_calibration_camera.py  # extrinsic solvers, synthetic
    python tests/test_liveview.py        # snapshot projection/drawing, and OFF
    python tests/test_hardware.py --sim  # hardware assumptions
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

The hardware world has two camera modes, chosen by `CAMERA_MODE` in
`src/config.py`: `'wrist'` (eye-in-hand, the default) and `'fixed'`
(eye-to-hand, for a camera on a stand). The wrist mode is the one the hardware
path is built and calibrated for; the fixed mode exists as a fallback for a
constrained cell and its detection path is not finished (`tests/test_fixed_camera.py`
is marked PENDING).

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
