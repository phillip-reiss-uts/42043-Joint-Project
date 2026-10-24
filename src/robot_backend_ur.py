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
identification (`servo_gains`). `tests/test_arm_control.py` is the
acceptance checklist; `tests/test_ur_backend.py` proves the plumbing
and `tests/test_gripper_backend.py` proves the RG2 client.
"""

import socket
import time

import numpy as np

from src import config as _cfg
from src.robot_backend import RobotBackend

UR_JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


class Robotiq2F85:
    # DEPRECATED -- WRONG DEVICE. This cell has an OnRobot RG2 v2, not a Robotiq
    # 2F-85; every Modbus read against the RG2 returns Illegal Data Address
    # (report 5.1). Nothing in the hardware path constructs this any more -- see
    # RG2 below. Kept only so the register-map test keeps exercising it until
    # review item todo E1 deletes both.
    #
    # Historical intent follows.
    #
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
        #
        # Two binding differences bite here, so both are absorbed rather than
        # assumed:
        #
        #   name   `servoJ` on the ur_rtde build used here; `servoj` in
        #          URScript and on older bindings.
        #   args   this build takes six POSITIONAL arguments only, and the
        #          order is NOT the natural one -- from the binding's own
        #          docstring:
        #              servoJ(q, speed, acceleration, time,
        #                     lookahead_time, gain)
        #          `speed` and `acceleration` are documented as "NOT used in
        #          current version", so they go in as 0. Their real slot is
        #          2nd and 3rd, which is why passing time there raises
        #          "not within [0.03;0.2]" (that range is lookahead_time's).
        #          gain's range is [100, 2000]; lookahead_time's is [0.03, 0.2].
        #
        # PACING. servoJ's docstring says it blocks for `time`, but this build
        # returns in ~0.1 ms and the arm keeps travelling afterwards. Measured
        # on the real UR3e (2026-10-23): 125 calls completed with a mean step of
        # 0.0 ms and the joint did not move at all, while ONE call moved it
        # 0.47 deg. So the cycle time is slept out here instead of assumed.
        # Without this, Arm.move's step count (dist / rate / dt) finishes before
        # the arm has moved, and the pick races ahead of the hardware.
        fn = getattr(self._control, 'servoJ', None) \
            or getattr(self._control, 'servoj')
        q = [float(v) for v in self._q_cmd]
        t_start = time.monotonic()
        try:
            fn(q, time=self._dt, lookahead_time=self.lookahead_time,
               gain=self.gain)
        except TypeError:
            fn(q, 0.0, 0.0, self._dt, self.lookahead_time, self.gain)
        self._gripper_send()
        rest = self._dt - (time.monotonic() - t_start)
        if rest > 0.0:
            time.sleep(rest)

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

    # ---- optional capability ----
    def grip_detected(self):
        # True/False if the gripper can say whether its jaws met an object,
        # None if it cannot. ADVISORY ONLY -- nothing branches on it, and it is
        # not part of the RobotBackend contract (that is why MujocoBackend has
        # no such method). The pick logs it as free evidence at the bench.
        fn = getattr(self._gripper, 'grip_detected', None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception:
            return None

    # ---- lifecycle ----
    def close(self):
        try:
            self._control.servoStop()
        except Exception:
            pass
        if self._gripper is not None:
            self._gripper.close()


# ---------------------------------------------------------------------------
# GRIPPER: OnRobot RG2 v2 -- what is actually on this cell
# ---------------------------------------------------------------------------
#
# CONTROL is the PROVEN route. Python triggers UR programs that were built on
# the pendant, over the dashboard server:
#
#     load /programs/rg_close.urp   ->   play
#
# It is DISCRETE: those programs open or close the jaws, they do not position
# them. So move() edge-triggers -- a 0..255 ramp fires one program at the
# crossing, not one program per control cycle.
#
# move() deliberately does NOT block. Waiting inside step() would stall servoJ
# streaming, and the caller already gives the jaws time: Arm.close_gripper ramps
# over STEPS_CLOSE cycles (24 s at 125 Hz) against a roughly 1 s stroke. If that
# ramp is ever shortened, set GRIP_STROKE_TIME_S and move() will wait the
# program out before returning.
#
# READBACK is the UNPROVEN route: jaw width comes from the URCap's XML-RPC on
# port 41414 via the `onRobot` package, never yet run on this cell. If it is
# absent or fails, state() returns NaN and the pipeline reads "unknown" -- which
# is what an open-loop pickup expects anyway.
#
# Quirks that cost time before (report 5.2):
#   * the load FILENAME is limited to ~15 characters on this controller
#   * one command per connection; replies get swallowed otherwise
#   * 'play' is refused unless the pendant is in Remote with the slider correct


def _dashboard(host, command, port=None, timeout=5.0):
    # One connection per command, by design: the controller swallows replies
    # when several commands share one connection.
    port = _cfg.UR_DASHBOARD_PORT if port is None else int(port)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((command + '\n').encode('ascii'))
        try:
            return sock.recv(1024).decode('utf-8', 'replace').strip()
        except socket.timeout:
            # A swallowed reply is not a failure; callers check for 'Failed'.
            return ''


# ---------------------------------------------------------------------------
# GRIPPER: OnRobot RG2 v2 -- what is actually on this cell
# ---------------------------------------------------------------------------
#
# PRIMARY: the OnRobot URCap's XML-RPC on port 41414, used exactly as the lab
# example does it (UR3_RG2_rosbridge/examples/15_real_gripper_control.py):
#
#     RG(host, grip_id).rg_grip(width_mm, force_N)      .get_rg_width()
#
# That route is PROPORTIONAL -- it commands a target jaw width in mm -- and it
# is the only one that reports width, which is why it comes first. Force is a
# write-only input there: nothing ever reads it back.
#
# LAST RESORT: trigger UR programs built on the pendant over the dashboard
# server (`load /programs/rg_close.urp` -> `play`). Discrete open/close only,
# no readback at all. Selected automatically, once, if XML-RPC is unavailable
# or a call fails.
#
# move() deliberately does NOT block on the XML-RPC route: the URCap returns
# when it has accepted the command. The caller already gives the jaws time --
# Arm.close_gripper ramps over STEPS_CLOSE cycles (24 s at 125 Hz) against a
# roughly 1 s stroke. If that ramp is shortened, set GRIP_STROKE_TIME_S and the
# dashboard fallback will wait the program out.
#
# Dashboard quirks that cost time before (report 5.2):
#   * the load FILENAME is limited to ~15 characters on this controller
#   * one command per connection; replies get swallowed otherwise
#   * 'play' is refused unless the pendant is in Remote with the slider correct


def _dashboard(host, command, port=None, timeout=5.0):
    # One connection per command, by design: the controller swallows replies
    # when several commands share one connection.
    port = _cfg.UR_DASHBOARD_PORT if port is None else int(port)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((command + '\n').encode('ascii'))
        try:
            return sock.recv(1024).decode('utf-8', 'replace').strip()
        except socket.timeout:
            # A swallowed reply is not a failure; callers check for 'Failed'.
            return ''


def _xmlrpc_call(host, method, *params):
    # The URCap speaks plain XML-RPC on RG2_XMLRPC_PORT. The onRobot package only
    # wraps a couple of methods, so anything else goes direct through stdlib --
    # no pycurl needed. Injected in tests, like the client and the dashboard.
    import xmlrpc.client
    uri = 'http://%s:%d' % (host, int(_cfg.RG2_XMLRPC_PORT))
    with xmlrpc.client.ServerProxy(uri) as proxy:
        return getattr(proxy, method)(*params)


class RG2:
    # OnRobot RG2 v2: XML-RPC control and readback, dashboard as last resort.
    #
    # Mirrors the lab example's calls: construct RG(host, id) once, then
    # rg_grip(width_mm, force_N) and get_rg_width().

    def __init__(self, host, grip_id=0, open_width_mm=None, close_width_mm=None,
                 force_n=None, min_step_mm=None, stroke_time=None,
                 dashboard_fallback=True, dashboard_timeout=5.0,
                 client=None, dashboard=None, call=None):
        self.host = host
        self._grip_id = int(grip_id)
        self.open_width_mm = (float(_cfg.RG2_OPEN_WIDTH_MM)
                              if open_width_mm is None else float(open_width_mm))
        self.close_width_mm = (float(_cfg.RG2_CLOSE_WIDTH_MM)
                               if close_width_mm is None
                               else float(close_width_mm))
        self.force_n = (float(_cfg.RG2_GRIP_FORCE_N)
                        if force_n is None else float(force_n))
        self.min_step_mm = (float(_cfg.RG2_MIN_STEP_MM)
                            if min_step_mm is None else float(min_step_mm))
        # Only the dashboard route needs a stroke time.
        self.stroke_time = (_cfg.GRIP_STROKE_TIME_S
                            if stroke_time is None else stroke_time)
        self._timeout = float(dashboard_timeout)
        self._allow_dashboard = bool(dashboard_fallback)
        # Injected for tests, the same way URBackend takes its RTDE clients.
        self._dashboard_fn = dashboard or _dashboard
        self._read_raw = call          # raw XML-RPC transport, lazily defaulted
        self._rg = client            # onRobot RG instance, or False once failed
        self._route = 'xmlrpc' if client is not None else None
        self._warned = False
        self._read_warned = False
        self._sent_width = None
        self._closed = None          # dashboard-route state
        self._last_q = None
        self._last_t = None

    # ---- route selection --------------------------------------------------
    def _xmlrpc(self):
        if self._rg is not None:
            return self._rg or None
        try:
            from onRobot.gripper import RG as _RG    # noqa: N813
            self._rg = _RG(self.host, self._grip_id)
            self._route = 'xmlrpc'
        except Exception as exc:
            self._rg = False
            self._route = 'dashboard' if self._allow_dashboard else None
            self._warn(exc)
        return self._rg or None

    def _warn(self, exc):
        if self._warned:
            return
        self._warned = True
        if self._route == 'dashboard':
            print('[HW] RG2 XML-RPC unavailable (%s); falling back to the '
                  'dashboard programs -- discrete open/close, no width' % exc)
        else:
            print('[HW] RG2 control unavailable (%s); the gripper will not '
                  'move and grasp state will read "unknown"' % exc)

    # ---- control ----------------------------------------------------------
    def _width_for(self, position):
        # GRIP_OPEN..GRIP_CLOSED (normally 0..255) -> jaw width in mm.
        lo, hi = float(_cfg.GRIP_OPEN), float(_cfg.GRIP_CLOSED)
        span = hi - lo
        frac = (0.0 if span <= 0.0
                else (min(max(float(position), lo), hi) - lo) / span)
        return self.open_width_mm + (self.close_width_mm - self.open_width_mm) * frac

    def move(self, position):
        if self._route is None and self._xmlrpc() is None:
            return
        if self._route == 'xmlrpc':
            self._move_xmlrpc(position)
        elif self._route == 'dashboard':
            self._move_dashboard(position)

    def _move_xmlrpc(self, position):
        width = self._width_for(position)
        if (self._sent_width is not None
                and abs(width - self._sent_width) < self.min_step_mm):
            return
        try:
            self._rg.rg_grip(width, self.force_n)
            self._sent_width = width
        except Exception as exc:
            # One failure is enough to stop trusting the route.
            self._rg = False
            self._route = 'dashboard' if self._allow_dashboard else None
            self._warn(exc)
            if self._route == 'dashboard':
                self._move_dashboard(position)

    def _move_dashboard(self, position):
        # Discrete jaws: act only when the command crosses the midpoint.
        mid = (float(_cfg.GRIP_OPEN) + float(_cfg.GRIP_CLOSED)) / 2.0
        want_closed = float(position) >= mid
        if self._closed is None:
            self._closed = not want_closed      # make the first command act
        if want_closed == self._closed:
            return
        self._run_program(_cfg.RG2_CLOSE_PROGRAM if want_closed
                          else _cfg.RG2_OPEN_PROGRAM)
        self._closed = want_closed

    def _run_program(self, program):
        reply = self._dashboard_fn('load %s' % program,
                                   _cfg.UR_DASHBOARD_PORT, self._timeout)
        if 'Failed' in reply:
            raise RuntimeError(
                'gripper program load failed (%r) -- the load filename is '
                'capped at ~15 characters on this controller' % reply)
        reply = self._dashboard_fn('play', _cfg.UR_DASHBOARD_PORT, self._timeout)
        if 'Failed' in reply:
            raise RuntimeError(
                'gripper play refused (%r) -- the pendant must be in Remote '
                'with the slider correct; a successful load does not imply '
                'play will work' % reply)
        if self.stroke_time:
            self._wait_done()

    def _wait_done(self):
        # Ask the pendant instead of sleeping blindly. Bounded either way.
        deadline = time.monotonic() + float(self.stroke_time)
        while time.monotonic() < deadline:
            state = self._dashboard_fn('programState', _cfg.UR_DASHBOARD_PORT,
                                       self._timeout)
            if 'PLAYING' not in state.upper():
                return
            time.sleep(0.05)

    # ---- readback ---------------------------------------------------------
    def _read_width(self):
        rg = self._xmlrpc()
        if rg is None:
            return None
        try:
            return float(rg.get_rg_width())
        except Exception as exc:
            # A failed READ does not invalidate CONTROL: rg_grip may still work,
            # so the route is left alone and only the readback is given up.
            if not self._read_warned:
                print('[HW] gripper width read failed (%s); grasp state will '
                      'read "unknown"' % exc)
                self._read_warned = True
            return None

    def grip_detected(self):
        # Optional extra: the URCap exposes rg_get_grip_detected, but the
        # onRobot package never wired it. Same envelope as rg_get_width -- one
        # int param -- so it goes direct through the raw XML-RPC transport.
        #
        # Returns True/False, or None if the URCap does not answer. Nothing
        # branches on it: the pickup is open loop, and this exists so the bench
        # can ask the gripper directly whether it is holding something.
        try:
            call = self._read_raw or (lambda m, *p: _xmlrpc_call(self.host, m, *p))
            return bool(call('rg_get_grip_detected', self._grip_id))
        except Exception:
            return None

    def state(self):
        # The (q, v, force) trio Arm.grasp_state consumes.
        #
        # q maps jaw OPENING onto the sim's driver-joint convention, where q
        # RISES as the jaws CLOSE; config explains the inversion.
        # force has no measurement at all -- the RG2 takes it as a write-only
        # input. NaN is honest and makes grasp_state say "unknown".
        width = self._read_width()
        if width is None:
            return {'q': float('nan'), 'v': float('nan'), 'force': float('nan')}
        width = min(max(width, 0.0), _cfg.GRIP_WIDTH_FULL_MM)
        q = _cfg.GRIP_Q_FULL * (1.0 - width / _cfg.GRIP_WIDTH_FULL_MM)
        now = time.monotonic()
        v = 0.0
        if self._last_q is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-6:
                v = (q - self._last_q) / dt
        self._last_q, self._last_t = q, now
        return {'q': float(q), 'v': float(v), 'force': float('nan')}

    def close(self):
        # No connection is held open and no program is left running.
        pass


def connect(ip, dt=0.008, lookahead_time=0.1, gain=300,
            activate_gripper=True, gripper_unit=None, grip_id=0,
            open_width_mm=None, close_width_mm=None, force_n=None,
            min_step_mm=None, stroke_time=None, dashboard_fallback=True):
    # Build a URBackend against a real controller.
    #
    # Only this function needs ur_rtde. The gripper needs nothing beyond the
    # lazily imported `onRobot` package, with dashboard sockets as last resort.
    #
    # gripper_unit and activate_gripper are kept for call-site compatibility:
    # they were the Robotiq 2F-85's Modbus unit and activation handshake.
    # gripper_unit is ignored now; activate_gripper still means "no gripper at
    # all", which is what run_hardware.py --no-gripper relies on.
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    control = RTDEControlInterface(ip)
    receive = RTDEReceiveInterface(ip)
    gripper = None
    if activate_gripper:
        gripper = RG2(ip, grip_id=grip_id, open_width_mm=open_width_mm,
                      close_width_mm=close_width_mm, force_n=force_n,
                      min_step_mm=min_step_mm, stroke_time=stroke_time,
                      dashboard_fallback=dashboard_fallback)
    return URBackend(control, receive, gripper, dt=dt,
                     lookahead_time=lookahead_time, gain=gain)
