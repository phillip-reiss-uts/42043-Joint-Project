"""UR3e backend: RTDE for the arm, XML-RPC for the OnRobot RG2.

WHAT THIS IS
------------
The hardware implementation of `RobotBackend` (src/robot_backend.py). It is the
counterpart of `MujocoBackend`: where that one writes `data.ctrl` and steps a
simulated clock, this one streams `servoj` targets to the UR controller and
reads the gripper's jaw width.

The robot layer never imports this module. `run_hardware.py` constructs it and
injects it into `Arm`, exactly as `run_simulation.py` injects `MujocoBackend`.

TESTABILITY
-----------
The two RTDE clients and the gripper are duck-typed constructor arguments, not
created inside the class, so `tests/test_ur_backend.py` can run the whole
backend against fakes with no robot present. `connect()` is the only place that
imports `ur_rtde`.

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

import contextlib
import io
import time

import numpy as np

@contextlib.contextmanager
def _quiet():
    # The lab's onRobot package prints the raw XML-RPC response body on EVERY
    # call (gripper.py, print(response.decode())), which floods the console with
    # <methodResponse> blocks once the loop sends a grip width per cycle.
    # Swallow stdout only around those calls; this module's own lines still
    # print normally.
    with contextlib.redirect_stdout(io.StringIO()):
        yield

from src import config as _cfg
from src.robot_backend import RobotBackend

UR_JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


class URBackend(RobotBackend):
    # RTDE joint streaming + RG2 width readback.

    def __init__(self, control, receive, gripper=None, n=6, dt=0.008,
                 lookahead_time=0.1, gain=300, gripper_period=0.05):
        # `control`  e.g. ur_rtde.RTDEControlInterface: needs .servoj(q, ...)
        # `receive`  e.g. ur_rtde.RTDEReceiveInterface: needs .getActualQ()
        # `gripper`  an RG2 (or any object with .move/.state)
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
        # Send the pending grip command, no faster than gripper_period.
        # _grip_sent tracks the command, not the send, so a throttled target is
        # not retried on the next cycle.
        if self._gripper is None:
            return
        target = int(round(np.clip(self._grip_cmd, 0.0, 255.0)))
        now = time.monotonic()
        if force or target != self._grip_sent:
            if (now - self._grip_t) >= self.gripper_period:
                self._gripper.move(target)
                self._grip_t = now
            self._grip_sent = target

    def step(self):
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
            return {'q': float('nan'), 'v': float('nan'), 'force': float('nan'),
                    'detected': None, 'width_mm': None}
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
        # Whether the jaws met an object: True/False, or None if the URCap does
        # not answer. Hardware only; reaches Arm.grasp_state via
        # gripper_state()['detected'].
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
# GRIPPER: OnRobot RG2 v2 -- XML-RPC control and readback
# ---------------------------------------------------------------------------
# rg_grip(width_mm, force_N) and get_rg_width() over the URCap on port 41414.
# Force is write-only; width is the only feedback.

def _xmlrpc_call(host, method, *params):
    # The URCap speaks plain XML-RPC on RG2_XMLRPC_PORT. The onRobot package only
    # wraps a couple of methods, so anything else goes direct through stdlib --
    # no pycurl needed. Injected in tests, like the client.
    import xmlrpc.client
    uri = 'http://%s:%d' % (host, int(_cfg.RG2_XMLRPC_PORT))
    with xmlrpc.client.ServerProxy(uri) as proxy:
        return getattr(proxy, method)(*params)


class RG2:
    # OnRobot RG2 v2: XML-RPC control and readback.
    #
    # Mirrors the lab example's calls: construct RG(host, id) once, then
    # rg_grip(width_mm, force_N) and get_rg_width().

    def __init__(self, host, grip_id=0, open_width_mm=None, close_width_mm=None,
                 force_n=None, min_step_mm=None, client=None, call=None):
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
        self._read_raw = call          # raw XML-RPC transport, lazily defaulted
        self._rg = client            # onRobot RG instance, or False once failed
        self._route = 'xmlrpc' if client is not None else None
        self._warned = False
        self._read_warned = False
        self._sent_width = None
        self._last_q = None
        self._last_t = None

    # ---- route selection --------------------------------------------------
    def probe(self):
        # Decide the route NOW and prove it answers. Called by connect() straight
        # after the RTDE interfaces come up.
        #
        # A CONSTRUCT IS NOT A CONNECTION. RG(host, id) only records the address
        # and builds an XML-RPC proxy; it never touches the device. Measured
        # against a dead host: the constructor succeeds and the route looks fine,
        # which is why a probe that only constructs is worthless. The width READ
        # is what proves the URCap is answering.
        #
        # Returns:
        #   'xmlrpc'         constructed and the width came back
        #   'xmlrpc-noread'  constructed, but the width read failed -- the URCap
        #                    is not answering, so writes will be refused too
        #   None             XML-RPC unavailable
        #
        # Note what this cannot test: whether a WRITE is accepted. That would
        # move the jaws, and the session requirement is about writes. The read is
        # the strongest check available without commanding the gripper.
        if self._route is None:
            self._xmlrpc()
        if self._route != 'xmlrpc':
            return self._route
        self.probe_width = self._read_width()
        return 'xmlrpc' if self.probe_width is not None else 'xmlrpc-noread'

    def _xmlrpc(self):
        if self._rg is not None:
            return self._rg or None
        try:
            from onRobot.gripper import RG as _RG    # noqa: N813
            with _quiet():
                self._rg = _RG(self.host, self._grip_id)
            self._route = 'xmlrpc'
        except Exception as exc:
            self._rg = False
            self._route = None
            self._warn(exc)
        return self._rg or None

    def _warn(self, exc):
        if self._warned:
            return
        self._warned = True
        print('[HW] RG2 control unavailable (%s); the gripper will not move '
              'and grasp state will read "unknown"' % exc)

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

    def _move_xmlrpc(self, position):
        width = self._width_for(position)
        if (self._sent_width is not None
                and abs(width - self._sent_width) < self.min_step_mm):
            return
        last = None
        # One retry, so a single timeout does not disable the route for the
        # session.
        for _attempt in (0, 1):
            try:
                with _quiet():
                    self._rg.rg_grip(width, self.force_n)
                self._sent_width = width
                return
            except Exception as exc:
                last = exc
        # Two failures: stop trusting the route.
        self._rg = False
        self._route = None
        self._warn(last)

    # ---- readback ---------------------------------------------------------
    def _read_width(self):
        rg = self._xmlrpc()
        if rg is None:
            return None
        try:
            with _quiet():
                width = float(rg.get_rg_width())
            return width
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
        # The (q, v, force, detected) quartet Arm.grasp_state consumes.
        #
        # q is jaw opening mapped onto the sim's driver-joint convention, so it
        # rises as the jaws close. force is unmeasured and always NaN: the RG2
        # takes it as a write-only input. detected is the URCap's contact bit,
        # read after the width so the two describe the same moment.
        width = self._read_width()
        detected = self.grip_detected()
        if width is None:
            return {'q': float('nan'), 'v': float('nan'), 'force': float('nan'),
                    'detected': detected, 'width_mm': None}
        width = min(max(width, 0.0), _cfg.GRIP_WIDTH_FULL_MM)
        q = _cfg.GRIP_Q_FULL * (1.0 - width / _cfg.GRIP_WIDTH_FULL_MM)
        now = time.monotonic()
        v = 0.0
        if self._last_q is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-6:
                v = (q - self._last_q) / dt
        self._last_q, self._last_t = q, now
        return {'q': float(q), 'v': float(v), 'force': float('nan'),
                'detected': detected, 'width_mm': float(width)}

    def close(self):
        # No connection is held open.
        pass


def connect(ip, dt=0.008, lookahead_time=0.1, gain=300,
            activate_gripper=True, grip_id=0,
            open_width_mm=None, close_width_mm=None, force_n=None,
            min_step_mm=None):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    control = RTDEControlInterface(ip)
    receive = RTDEReceiveInterface(ip)
    gripper = None
    if activate_gripper:
        gripper = RG2(ip, grip_id=grip_id, open_width_mm=open_width_mm,
                      close_width_mm=close_width_mm, force_n=force_n,
                      min_step_mm=min_step_mm)
        route = gripper.probe()
        if route == 'xmlrpc':
            print('[HW] gripper: XML-RPC answered, width %.1f mm'
                  % gripper.probe_width)
        elif route == 'xmlrpc-noread':
            print('[HW] gripper: constructed but the width read FAILED')
            print('     the URCap is not answering, so rg_grip will be refused: '
                  'check it is running in Tool/IO')
        else:
            print('[HW] gripper: NO ROUTE (XML-RPC unavailable)')
            print('     grasps will report unknown and never squeeze')
    return URBackend(control, receive, gripper, dt=dt,
                     lookahead_time=lookahead_time, gain=gain)
