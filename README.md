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

Simulation keys: `6` spawn, `7` pick up / put down (toggle), `8` pause,
`9` reset, `.` object speed. On macOS use `mjpython main.py`.

`--lightweight` drops the RGB-image and point-cloud panels. Each is its own GL
render on top of the depth pass perception already does, so on a machine without
a decent GPU they dominate the frame and a pick takes minutes of wall time
instead of seconds. The 3D scene, the tool trail, the detections and the text
panels all stay — the robot uses none of the panels.

## Camera calibration

`CAM_POS`/`CAM_QUAT` in `src/config.py` are the simulator's bracket, not a
measurement. Every detected centre inherits that error directly, so calibrate
before a hardware run. Print `models/aruco/marker0_DICT_4X4_50_100mm.pdf`, lay
it flat on the table, then:

    python tools/calibrate_camera.py --wrist

The marker size defaults to 0.10 m, which is that file. If you print a different
size, pass `--marker-size` with the edge you actually measured.

Print at **100% / actual size, not "fit to page"**: the page is 140 mm and the
black square 100 mm, so scaling up by a quarter gives a 125 mm marker while the
tool still assumes 0.10 — a silent 25% error in every number that follows.
Measure the printed square before trusting it.

It captures the marker once and asks you to touch two adjacent corners with the
TCP, which fixes the board's pose in the base frame. Then move the arm to
`--poses` poses (default 12), twisting the wrist between them so the mount's
rotation is actually constrained. Each pose gives one estimate of the same
constant; the tool prints the mean with its error bar. Sample output only:

    RESULT  (wrist, 12 poses)  spread 1.8 mm / 0.6 deg
    CAM_POS  = (0.04821, -0.00312, 0.08764)
    CAM_QUAT = (0.501204, -0.498871, 0.498871, 0.501204)     # w, x, y, z

Paste both into `src/config.py`. `CAM_MOUNT_POS`/`_QUAT` alias them, so the
simulator and the hardware move together; leave `HW_CAM_DELTA_*` at identity,
since this is the measurement that delta exists for.

The spread is the result, not the average. Under 5 mm / 2 deg means one constant
describes every pose, and the tool exits 0. Well above that means the clamp
flexes, the marker moved, or a pose was taken with the marker half out of frame
— re-run rather than trusting the mean. Then check it against a ruler:

    python tests/test_camera_ruler.py --truth 0.0,-0.27

A constant bias across three table spots is the mount; an error that grows
across the table is the extrinsic or a scale mistake.

A camera on a stand instead of the wrist (`CAMERA_MODE = 'fixed'`) is the
fallback. Its two modes print `FIXED_CAM_POS/TARGET/UP` rather than the mount:

    python tools/calibrate_camera.py --touch     # robot touches the cube
    python tools/calibrate_camera.py --aruco     # one marker image

## Tests

    python tests/test_layering.py        # robot/world + provider separation
    python tests/test_task_flow.py       # pickup toggle: grasp -> hold -> put down
    python tests/test_dynamics.py        # analytic M(q)/g(q) vs MuJoCo
    python tests/test_ur_kinematics.py   # hardware chain vs MuJoCo
    python tests/test_kinematics_parity.py  # TCP pose + frame convention
    python tests/test_ur_backend.py      # RTDE/RG2 plumbing, no robot
    python tests/test_fixed_camera.py    # eye-to-hand world, offline (PENDING)
    python tests/test_calibration_camera.py  # extrinsic solvers, synthetic
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
