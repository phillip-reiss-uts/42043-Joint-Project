# Vision-Guided Pick (UR3e + 2F-85 + D435i)

A UR3e with a wrist-mounted RealSense D435i and a Robotiq 2F-85 that tracks a
cube, grasps it on command, and holds it until told to put it down. Runs in
MuJoCo and on the real robot from the same robot code.

## Setup

    python -m pip install -r requirements.txt

`pyrealsense2` is hardware-only and is not in `requirements.txt`.

## Run

    python main.py                # simulation (default)
    python main.py --hardware --ip 192.168.1.10

Simulation keys: `6` spawn, `7` pick up / put down (toggle), `8` pause,
`9` reset, `.` object speed. On macOS use `mjpython main.py`.

## Tests

    python tests/test_layering.py        # robot/world + provider separation
    python tests/test_task_flow.py       # pickup toggle: grasp -> hold -> put down
    python tests/test_dynamics.py        # analytic M(q)/g(q) vs MuJoCo
    python tests/test_ur_kinematics.py   # hardware chain vs MuJoCo
    python tests/test_ur_backend.py      # RTDE/Modbus plumbing, no robot
    python tests/test_hardware.py --sim  # hardware assumptions

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

## Full notes

The detailed document — control design, measurements, the gravity and
integrator history, grasp accuracy, and the hardware porting checklist — lives
one level up as [README-assignment4-full.md](../README-assignment4-full.md).
