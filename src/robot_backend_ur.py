"""UR3e backend: RTDE for the arm, Modbus for the Robotiq 2F-85.

WHAT THIS IS
------------
The hardware implementation of `RobotBackend` (src/robot_backend.py). It is the
counterpart of `MujocoBackend`: where that one writes `data.ctrl` and steps a
simulated clock, this one streams `servoj` targets to the UR controller and
reads the gripper's registers.

The robot layer never imports this module. `run_hardware.py` constructs it and
injects it into `Arm`, exactly as `run_simulation.py` injects `MujocoBackend`.

TESTABILITY
-----------
The two RTDE clients and the gripper are duck-typed constructor arguments, not
created inside the class, so `tests/test_ur_backend.py` can run the whole
backend against fakes with no robot present. `connect()` is the only place that
imports `ur_rtde` and `pymodbus`.

GRAVITY COMPENSATION
--------------------
None. A UR controller already compensates the arm's weight in its own servo
loop, which is precisely what `MujocoBackend.gravity_feedforward` emulates. The
contract is "each backend supplies what its controller needs, exactly once" --
adding a second term here would reintroduce the upward bias fixed in the
simulation.

UNVERIFIED
----------
This module has never run against a robot. Every hardware-specific number is
either taken from a published document (register map) or marked as needing
identification (`servo_gains`). `tests/test_hardware.py --real` is the
acceptance checklist; `tests/test_ur_backend.py` proves the plumbing.
"""

import time

import numpy as np

from src.robot_backend import RobotBackend

UR_JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


class Robotiq2F85:
    # The Robotiq 2F-85 over Modbus RTU/TCP.
    #
    # Register map from the 2F-85 instruction manual (0-based holding
    # registers). The three the robot consumes are the same trio the simulation
    # reads from its finger joints: position, velocity, force.

    # command registers
    ACTION = 0x03E8        # rACT: 0 = reset, 1 = activate
    GOTO = 0x03E9          # rGTO: 0 = stop, 1 = go to position
    POSITION = 0x03EA      # rPR: 0 (open) .. 255 (closed)
    SPEED = 0x03EB         # rSP
    FORCE = 0x03EC         # rFR
    # status registers
    gOBJ = 0x07D0          # grasp detected
    gSTA = 0x07D1
    gGTO = 0x07D2
    gACT = 0x07D3
    gPO = 0x07D7           # actual finger position, 0 .. 255
    gCU = 0x07D8           # motor current, 0 .. 255

    Q_SCALE = 0.8          # rad-equivalent range the sim's driver joints span
    F_SCALE = 5.0          # N-equivalent force range the sim uses

    def __init__(self, client, unit=9, speed=255, force=255):
        self.client = client
        self.unit = int(unit)
        self.speed = int(speed)
        self.force_limit = int(force)
        self._last_q = None
        self._last_t = None

    # ---- low level ----
    def _write(self, addr, value):
        # pymodbus 3.x renamed `unit` to `slave`; accept either.
        try:
            return self.client.write_register(addr, int(value), slave=self.unit)
        except TypeError:
            return self.client.write_register(addr, int(value), unit=self.unit)

    def _read(self, addr, count=1):
        try:
            rr = self.client.read_holding_registers(addr, count, slave=self.unit)
        except TypeError:
            rr = self.client.read_holding_registers(addr, count, unit=self.unit)
        if rr is None or getattr(rr, 'isError', lambda: False)():
            return None
        return list(rr.registers)

    def _read_one(self, addr):
        regs = self._read(addr, 1)
        return None if not regs else int(regs[0])

    # ---- setup ----
    def activate(self, timeout=10.0):
        # rACT=1, then wait for gACT=1. Returns True when the gripper is ready.
        self._write(self.ACTION, 1)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._read_one(self.gACT) == 1:
                self._write(self.GOTO, 1)
                return True
            time.sleep(0.05)
        return False

    def move(self, position):
        # Command a finger position: 0 = open, 255 = closed.
        self._write(self.SPEED, self.speed)
        self._write(self.FORCE, self.force_limit)
        self._write(self.GOTO, 1)
        self._write(self.POSITION, int(np.clip(position, 0, 255)))

    # ---- readback ----
    def state(self):
        # The (q, v, force) trio Arm.grasp_state consumes.
        #
        # The scaling maps the device's 0..255 registers onto the simulation's
        # units so the SAME thresholds apply. It is a monotone placeholder, not
        # a calibration: gCU is a motor current, and turning it into newtons
        # needs the gripper's force constant. --real checks the two endpoints.
        po = self._read_one(self.gPO)
        cu = self._read_one(self.gCU)
        if po is None or cu is None:
            return {'q': float('nan'), 'v': float('nan'), 'force': float('nan')}

        q = self.Q_SCALE * po / 255.0
        now = time.monotonic()
        v = 0.0
        if self._last_q is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-6:
                v = (q - self._last_q) / dt
        self._last_q, self._last_t = q, now
        return {'q': float(q), 'v': float(v),
                'force': float(self.F_SCALE * cu / 255.0)}

    def close(self):
        # Best effort: stop the motion and drop the connection.
        try:
            self._write(self.GOTO, 0)
        finally:
            try:
                self.client.close()
            except Exception:
                pass


class URBackend(RobotBackend):
    # RTDE joint streaming + Robotiq readback.

    def __init__(self, control, receive, gripper=None, n=6, dt=0.008,
                 lookahead_time=0.1, gain=300, gripper_period=0.05):
        # `control`  e.g. ur_rtde.RTDEControlInterface: needs .servoj(q, ...)
        # `receive`  e.g. ur_rtde.RTDEReceiveInterface: needs .getActualQ()
        # `gripper`  a Robotiq2F85 (or any object with .move/.state)
        self._control = control
        self._receive = receive
        self._gripper = gripper
        self._n = int(n)
        self._dt = float(dt)
        self.lookahead_time = float(lookahead_time)
        self.gain = float(gain)

        self._q_cmd = np.array(receive.getActualQ(), float)[:self._n]
        self._grip_cmd = 0.0
        self._grip_sent = None
        self._grip_t = -1e9
        self.gripper_period = float(gripper_period)
        self._t0 = time.monotonic()

    # ---- state ----
    def joint_positions(self):
        return np.array(self._receive.getActualQ(), float)[:self._n]

    def joint_velocities(self):
        qd = getattr(self._receive, 'getActualQd', None)
        if qd is None:
            return np.zeros(self._n)
        return np.array(qd(), float)[:self._n]

    def time(self):
        # Controller timestamp when available (RTDE reports one); otherwise a
        # monotonic wall clock. Both are monotonic, which is all the servo's
        # filters need.
        ts = getattr(self._receive, 'getTimestamp', None)
        if ts is not None:
            try:
                return float(ts())
            except Exception:
                pass
        return time.monotonic() - self._t0

    @property
    def dt(self):
        return self._dt

    # ---- commands ----
    def command(self, q_arm, grip):
        # Store the targets; nothing is sent until step(), mirroring
        # MujocoBackend writing data.ctrl.
        self._q_cmd = np.asarray(q_arm, float)[:self._n].copy()
        self._grip_cmd = float(grip)

    def _gripper_send(self, force=False):
        # Throttle Modbus writes: the close/open ramps change the command every
        # control cycle, and the 2F-85 must not be written at 125 Hz.
        if self._gripper is None:
            return
        target = int(round(np.clip(self._grip_cmd, 0.0, 255.0)))
        now = time.monotonic()
        if (force or target != self._grip_sent) \
                and (now - self._grip_t) >= self.gripper_period:
            self._gripper.move(target)
            self._grip_sent = target
            self._grip_t = now

    def step(self):
        # One control cycle: stream the target and block for the cycle time.
        # servoj's time argument is the period the controller should take.
        self._control.servoj([float(v) for v in self._q_cmd],
                             time=self._dt,
                             lookahead_time=self.lookahead_time,
                             gain=self.gain)
        self._gripper_send()

    # ---- gripper readback ----
    def gripper_state(self):
        if self._gripper is None:
            return {'q': float('nan'), 'v': float('nan'), 'force': float('nan')}
        return self._gripper.state()

    # ---- model handles ----
    def joint_addresses(self, name):
        # A UR controller has no qpos array: the "address" of a joint is its
        # index in the RTDE joint vector. The robot layer only uses this to
        # resolve names, so an index pair is a faithful stand-in.
        if name in UR_JOINT_NAMES:
            i = UR_JOINT_NAMES.index(name)
            return i, i
        return None

    @property
    def tool_body_id(self):
        # There is no body id on hardware. The robot layer uses this only where
        # a simulator body would be needed, and the hardware front-end reads the
        # tool pose through ObjectWorld.gripper_pose() instead.
        return -1

    # ---- lifecycle ----
    def close(self):
        try:
            self._control.servoStop()
        except Exception:
            pass
        if self._gripper is not None:
            self._gripper.close()


def connect(ip, gripper_unit=9, dt=0.008, lookahead_time=0.1, gain=300,
            activate_gripper=True):
    # Build a URBackend against a real controller.
    #
    # This is the only function that needs ur_rtde and pymodbus installed.
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    from pymodbus.client import ModbusTcpClient

    control = RTDEControlInterface(ip)
    receive = RTDEReceiveInterface(ip)
    client = ModbusTcpClient(ip, port=502)
    gripper = Robotiq2F85(client, unit=gripper_unit)
    if activate_gripper and not gripper.activate():
        print('[HW] gripper did not report activated; grasp checks will say '
              '"unknown"')
    return URBackend(control, receive, gripper, dt=dt,
                     lookahead_time=lookahead_time, gain=gain)
