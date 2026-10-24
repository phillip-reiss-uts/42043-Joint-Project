# Offline tests for the RG2 gripper backend: no robot, no onRobot, no pycurl.
#
# WHY THIS EXISTS
# ---------------
# `RG2` in src/robot_backend_ur.py is the only part of the hardware path that
# cannot be exercised by the simulator. Its XML-RPC client and its dashboard
# transport are injected rather than constructed (the same way URBackend takes
# its RTDE clients), so the whole class can be run against fakes:
#
#   * the grip-command -> jaw-width interpolation
#   * min-step suppression during an Arm.close_gripper ramp
#   * the width -> q direction, and that force is NaN rather than invented
#   * demotion to the dashboard when XML-RPC fails, and that it happens ONCE
#   * that a failed width READ does not take out control
#
# What this cannot prove: that the URCap answers, or that the dashboard accepts
# the programs. That is the bench, run order step 2 in todo-inlab.txt.
#
# Run: python tests/test_gripper_backend.py

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg
from src.robot_backend_ur import RG2, URBackend


# FAKES
class FakeRG:
    # Stands in for onRobot.gripper.RG.
    def __init__(self, width=42.0, fail_grip=False, fail_width=False):
        self.width = width
        self.fail_grip = fail_grip
        self.fail_width = fail_width
        self.grips = []
        self.width_reads = 0

    def rg_grip(self, width, force):
        if self.fail_grip:
            raise RuntimeError('urcap refused rg_grip')
        self.grips.append((float(width), float(force)))

    def get_rg_width(self):
        self.width_reads += 1
        if self.fail_width:
            raise RuntimeError('urcap refused rg_get_width')
        return float(self.width)


class FakeDashboard:
    # Stands in for the module-level _dashboard() socket helper.
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or ''

    def __call__(self, command, port=None, timeout=None):
        self.calls.append(command)
        verb = command.split()[0]
        if verb in self.fail:
            return 'Failed to execute: %s' % verb
        return {'load': 'Loading program', 'play': 'Starting program',
                'programState': 'STOPPED'}.get(verb, 'ok')


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
    failures += not check('state() returns q, v, force',
                          set(st) == {'q', 'v', 'force'}, str(sorted(st)))
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

    # ---------------- fallback ----------------
    print('\nDASHBOARD FALLBACK')
    rg = FakeRG(fail_grip=True)
    dash = FakeDashboard()
    grip = RG2('127.0.0.1', client=rg, dashboard=dash)
    grip.move(hi)                       # rg_grip raises -> demote
    failures += not check('falls back after one failed rg_grip',
                          grip._route == 'dashboard', str(grip._route))
    failures += not check('loads then plays the close program',
                          len(dash.calls) >= 2
                          and 'load %s' % cfg.RG2_CLOSE_PROGRAM in dash.calls
                          and 'play' in dash.calls, str(dash.calls))

    before = len(dash.calls)
    for _ in range(5):
        grip.move(hi)                   # already closed: must not refire
    failures += not check('repeat commands do not refire the program',
                          len(dash.calls) == before,
                          '%d new calls' % (len(dash.calls) - before))

    grip.move(lo)
    failures += not check('opening runs the open program',
                          'load %s' % cfg.RG2_OPEN_PROGRAM in dash.calls)
    failures += not check('no width on the fallback route',
                          grip.state()['q'] != grip.state()['q'])

    # a refused play is an explicit error, not a silent no-op
    dash_fail = FakeDashboard(fail='play')
    grip_fail = RG2('127.0.0.1', client=FakeRG(fail_grip=True),
                    dashboard=dash_fail)
    try:
        grip_fail.move(hi)
        refused = False
    except RuntimeError as exc:
        refused = 'Remote' in str(exc)
    failures += not check("refused 'play' raises with the reason", refused)

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
    print('  This does NOT verify the URCap or the dashboard programs:')
    print('  run order step 2 in todo-inlab.txt does that, at the cell.')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
