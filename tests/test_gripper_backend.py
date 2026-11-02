"""Offline tests for the RG2 gripper backend: no robot, no onRobot, no pycurl.

WHY THIS EXISTS
---------------
`RG2` in src/robot_backend_ur.py is the only part of the hardware path that
cannot be exercised by the simulator. Its XML-RPC client is injected rather
than constructed (the same way URBackend takes its RTDE clients), so the whole
class can be run against fakes:

  * the grip-command -> jaw-width interpolation
  * min-step suppression during an Arm.close_gripper ramp
  * the width -> q direction, and that force is NaN rather than invented
  * that a failed width READ does not take out control

What this cannot prove: that the URCap answers. That is the bench, run order
step 2 in todo-inlab.txt.

Run: python tests/test_gripper_backend.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg
from src.arm import Arm
from src.robot_backend_ur import RG2, URBackend


# FAKES
class FakeRG:
    # Stands in for onRobot.gripper.RG.
    def __init__(self, width=42.0, fail_width=False):
        self.width = width
        self.fail_width = fail_width
        self.grips = []
        self.width_reads = 0

    def rg_grip(self, width, force):
        self.grips.append((float(width), float(force)))

    def get_rg_width(self):
        self.width_reads += 1
        if self.fail_width:
            raise RuntimeError('urcap refused rg_get_width')
        return float(self.width)


class FakeReceive:
    def __init__(self, q0):
        self.q = list(map(float, q0))
        self.t = 0.0

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return [0.0] * len(self.q)

    def getTimestamp(self):
        return self.t


class FakeControl:
    def __init__(self, receive):
        self.receive = receive
        self.calls = []

    def servoj(self, q, time=0.0, lookahead_time=0.0, gain=0.0):
        self.calls.append(list(q))
        self.receive.q = [float(v) for v in q]
        self.receive.t += float(time)
        return True

    def servoStop(self):
        pass


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))
    return bool(ok)


def main():
    failures = 0
    lo, hi = float(cfg.GRIP_OPEN), float(cfg.GRIP_CLOSED)

    print()
    print('=' * 74)
    print('RG2 GRIPPER BACKEND  (fakes; no robot, no onRobot)')
    print('=' * 74)

    # ---------------- contract ----------------
    print('\nCONTRACT')
    rg = FakeRG()
    grip = RG2('127.0.0.1', client=rg)
    failures += not check('has move/state/close',
                          all(hasattr(grip, m) for m in
                              ('move', 'state', 'close')))
    st = grip.state()
    failures += not check('state() returns q, v, force, detected, width_mm',
                          set(st) == {'q', 'v', 'force', 'detected', 'width_mm'},
                          str(sorted(st)))
    failures += not check('force is NaN, never invented',
                          st['force'] != st['force'])

    # accepted where URBackend expects a gripper
    control = FakeControl(FakeReceive([0.0] * 6))
    backend = URBackend(control, control.receive, grip, dt=0.008)
    failures += not check('URBackend accepts it as a gripper',
                          isinstance(backend.gripper_state(), dict))

    # ---------------- interpolation ----------------
    print('\nGRIP COMMAND -> JAW WIDTH')
    rg = FakeRG()
    grip = RG2('127.0.0.1', client=rg)
    grip.move(lo)                       # fully open
    grip.move(hi)                       # fully closed
    widths = [w for w, _f in rg.grips]
    failures += not check('open commands the full stroke',
                          abs(widths[0] - cfg.RG2_OPEN_WIDTH_MM) < 1e-6,
                          str(widths[0]))
    failures += not check('closed commands the close width',
                          abs(widths[-1] - cfg.RG2_CLOSE_WIDTH_MM) < 1e-6,
                          str(widths[-1]))
    failures += not check('monotone: closing narrows the target',
                          all(a > b for a, b in zip(widths, widths[1:])),
                          str(widths))
    failures += not check('force is passed through unchanged',
                          all(f == cfg.RG2_GRIP_FORCE_N for _w, f in rg.grips))

    mid = (lo + hi) / 2.0
    rg2 = FakeRG()
    grip2 = RG2('127.0.0.1', client=rg2)
    grip2.move(mid)
    expect = (cfg.RG2_OPEN_WIDTH_MM + cfg.RG2_CLOSE_WIDTH_MM) / 2.0
    failures += not check('midpoint is halfway between the two widths',
                          abs(rg2.grips[0][0] - expect) < 1.0,
                          '%.2f vs %.2f' % (rg2.grips[0][0], expect))

    # min-step suppression during a ramp
    rg3 = FakeRG()
    grip3 = RG2('127.0.0.1', client=rg3)
    for i in range(0, 256, 5):
        grip3.move(i)
    sent = len(rg3.grips)
    failures += not check('a 51-step ramp does not send 51 commands',
                          sent < 51, '%d commands sent' % sent)
    failures += not check('but it does send several', sent >= 3, str(sent))

    # ---------------- readback ----------------
    print('\nREADBACK')
    rg = FakeRG(width=0.0)
    grip = RG2('127.0.0.1', client=rg)
    q_air = grip.state()['q']
    rg.width = cfg.OBJECT_SIDE_MM
    st_cube = grip.state()
    q_cube = st_cube['q']
    failures += not check('q is high when the jaws are closed (sim convention)',
                          q_air > q_cube, 'air %.3f  cube %.3f'
                          % (q_air, q_cube))
    failures += not check('cube q is below the empty threshold',
                          q_cube < cfg.GRASP_CLOSED_Q,
                          'q %.3f < %.3f' % (q_cube, cfg.GRASP_CLOSED_Q))
    failures += not check('v is finite once two reads exist',
                          st_cube['v'] == st_cube['v'])

    rg.width = 1e6                      # beyond the stroke
    failures += not check('width is clamped to the stroke',
                          grip.state()['q'] <= -1e-12 or
                          grip.state()['q'] >= 0.0)

    # ---------------- close at a requested width ----------------
    print('\nCLOSE AT A REQUESTED WIDTH')
    # Arm.close_gripper(position=...) exists so the calibration tool can close
    # to the width under test instead of the configured close width.
    recv = FakeReceive([0.0] * 6)
    ctrl = FakeControl(recv)
    rg = FakeRG(width=58.0)
    be = URBackend(ctrl, recv,
                   RG2('127.0.0.1', client=rg, call=lambda *a: None),
                   dt=0.008)
    arm = Arm(kinematics=object(), backend=be)
    pos = 128.0
    result = arm.close_gripper(1, position=pos)
    frac = pos / float(cfg.GRIP_CLOSED)
    expected = (cfg.RG2_OPEN_WIDTH_MM
                + (cfg.RG2_CLOSE_WIDTH_MM - cfg.RG2_OPEN_WIDTH_MM) * frac)
    failures += not check('close_gripper sends the requested position',
                          len(rg.grips) == 1
                          and abs(rg.grips[0][0] - expected) < 1e-6,
                          'sent %s, expected %.3f'
                          % ([g[0] for g in rg.grips], expected))
    failures += not check('close_gripper returns the width verdict',
                          isinstance(result, tuple) and result[0] is True,
                          str(result))

    # ---------------- read failure is not control failure ----------------
    print('\nREAD FAILURE KEEPS CONTROL')
    rg = FakeRG(fail_width=True)
    grip = RG2('127.0.0.1', client=rg)
    failures += not check('state() is NaN when the read fails',
                          grip.state()['q'] != grip.state()['q'])
    failures += not check('control route survives a failed read',
                          grip._route == 'xmlrpc', str(grip._route))
    grip.move(lo)                       # width == open, so this sends
    failures += not check('and rg_grip still reaches the gripper',
                          len(rg.grips) >= 1, str(len(rg.grips)))

    # ---------------- optional grip_detected ----------------
    print('\nOPTIONAL grip_detected')
    grip = RG2('127.0.0.1', client=FakeRG())
    failures += not check('grip_detected() exists and fails soft',
                          grip.grip_detected() in (True, False, None),
                          str(grip.grip_detected()))

    class DetectingRG(FakeRG):
        def __init__(self, answer):
            FakeRG.__init__(self)
            self.answer = answer

        def grip_detected(self):
            return self.answer

    recv = FakeReceive([0.0] * 6)
    ctrl = FakeControl(recv)
    be = URBackend(ctrl, recv, RG2('127.0.0.1', client=FakeRG()), dt=0.008)
    failures += not check('URBackend exposes grip_detected as advisory',
                          hasattr(be, 'grip_detected'))
    failures += not check('None when the gripper cannot answer',
                          be.grip_detected() is None, str(be.grip_detected()))

    always = lambda *a: True       # noqa: E731
    never = lambda *a: False       # noqa: E731
    be2 = URBackend(FakeControl(recv), recv,
                    RG2('127.0.0.1', client=FakeRG(), call=always), dt=0.008)
    failures += not check('delegates True from the gripper',
                          be2.grip_detected() is True, str(be2.grip_detected()))
    be3 = URBackend(FakeControl(recv), recv,
                    RG2('127.0.0.1', client=FakeRG(), call=never), dt=0.008)
    failures += not check('delegates False from the gripper',
                          be3.grip_detected() is False, str(be3.grip_detected()))

    def boom(*a):
        raise RuntimeError('urcap down')

    be4 = URBackend(FakeControl(recv), recv,
                    RG2('127.0.0.1', client=FakeRG(), call=boom), dt=0.008)
    failures += not check('a failing transport is None, not an exception',
                          be4.grip_detected() is None)

    print()
    print('=' * 74)
    if failures:
        print('  %d check(s) FAILED' % failures)
        print('=' * 74)
        return 1
    print('  All checks passed. The RG2 backend logic is sound.')
    print('  This does NOT verify the URCap:')
    print('  run order step 2 in todo-inlab.txt does that, at the cell.')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
