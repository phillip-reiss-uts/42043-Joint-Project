# Offline tests for the hardware backend: no robot, no ur_rtde.
#
# WHY THIS EXISTS
# ---------------
# `src/robot_backend_ur.py` is the one part of the port that cannot be exercised
# by the simulator. Everything it does can still be tested, because its RTDE and
# gripper dependencies are injected rather than constructed: the fakes below
# stand in for the controller and the gripper, and the assertions are the
# RobotBackend contract.
#
# What this proves: the plumbing (servoj pacing, joint readback, the gripper
# command and its throttle, the grasp trio) and that the hardware front-end
# imports without the simulator. What it cannot prove: that the real controller
# accepts these calls -- that is `tests/test_arm_control.py`.
#
# Run: python tests/test_ur_backend.py

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.robot_backend import RobotBackend
from src.robot_backend_ur import URBackend


# FAKES
class FakeReceive:
    def __init__(self, q0):
        self.q = list(map(float, q0))
        self.qd = [0.0] * len(self.q)
        self.t = 0.0
        self.qd_calls = 0

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        self.qd_calls += 1
        return list(self.qd)

    def getTimestamp(self):
        return self.t


class FakeControl:
    # Records servoj calls and models the controller as "reaches the target".
    def __init__(self, receive):
        self.receive = receive
        self.calls = []
        self.times = []

    def servoj(self, q, time=0.0, lookahead_time=0.0, gain=0.0):
        self.calls.append(np.array(q, float))
        self.times.append(float(time))
        self.receive.q = [float(v) for v in q]
        self.receive.t += float(time)
        return True

    def servoStop(self):
        self.stopped = True


class FakeGripper:
    def __init__(self, state=None):
        self.moves = []
        self._state = state or {'q': 0.25, 'v': 0.0, 'force': 5.0}

    def move(self, position):
        self.moves.append(int(position))

    def state(self):
        return dict(self._state)

    def close(self):
        pass


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))
    return bool(ok)


def main():
    failures = 0
    q0 = [0.1, -1.2, 0.9, -1.0, -1.5, 0.2]

    print()
    print('=' * 74)
    print('HARDWARE BACKEND  (fakes; no robot, no ur_rtde)')
    print('=' * 74)

    # ---------------- contract ----------------
    print('\nCONTRACT')
    receive = FakeReceive(q0)
    control = FakeControl(receive)
    grip = FakeGripper()
    be = URBackend(control, receive, grip, dt=0.008)

    failures += not check('is a RobotBackend', isinstance(be, RobotBackend))
    failures += not check(
        'joint positions match the controller',
        np.allclose(be.joint_positions(), q0), str(np.round(be.joint_positions(), 3)))
    failures += not check('joint velocities have the right length',
                          be.joint_velocities().shape == (6,))
    failures += not check('dt is the control period', abs(be.dt - 0.008) < 1e-12)
    failures += not check('joint_addresses resolves a UR joint name',
                          be.joint_addresses('elbow_joint') == (2, 2))
    failures += not check('joint_addresses rejects an unknown name',
                          be.joint_addresses('gripper/base') is None)
    failures += not check('tool_body_id is the hardware sentinel',
                          be.tool_body_id == -1)
    failures += not check('no gravity feedforward is added (the controller does it)',
                          not hasattr(be, 'gravity_feedforward')
                          and not hasattr(be, 'qfrc_applied'))

    # ---------------- command / step ----------------
    print('\nCOMMAND AND STEP')
    cmd = np.array([0.2, -1.0, 0.8, -0.9, -1.4, 0.3])
    for _ in range(5):
        be.command(cmd, 255.0)
        be.step()
    failures += not check('one servoj per step', len(control.calls) == 5,
                          'calls=%d' % len(control.calls))
    failures += not check('the commanded joint vector is what is streamed',
                          all(np.allclose(c, cmd) for c in control.calls))
    failures += not check('the cycle time is passed to servoj',
                          all(abs(t - 0.008) < 1e-12 for t in control.times))
    failures += not check('time advances with the cycle',
                          abs(be.time() - 5 * 0.008) < 1e-9,
                          't=%.4f' % be.time())

    # ---------------- gripper throttle ----------------
    print('\nGRIPPER COMMAND')
    grip2 = FakeGripper()
    recv2 = FakeReceive(q0)
    ctrl2 = FakeControl(recv2)
    be2 = URBackend(ctrl2, recv2, grip2, dt=0.008, gripper_period=0.0)
    for i in range(11):
        be2.command(cmd, 255.0 * i / 10.0)
        be2.step()
    failures += not check('a full open->close ramp reaches the gripper',
                          len(grip2.moves) == 11 and grip2.moves[-1] == 255,
                          'moves=%s' % grip2.moves)
    failures += not check('the ramp is monotone',
                          grip2.moves == sorted(grip2.moves))

    grip3 = FakeGripper()
    be3 = URBackend(FakeControl(FakeReceive(q0)), FakeReceive(q0), grip3,
                    dt=0.008, gripper_period=1e9)
    for i in range(11):
        be3.command(cmd, 255.0 * i / 10.0)
        be3.step()
    failures += not check('writes are throttled (not one per control cycle)',
                          len(grip3.moves) == 1, 'moves=%d' % len(grip3.moves))

    # ---------------- grasp readback ----------------
    print('\nGRASP READBACK')
    gs = be.gripper_state()
    failures += not check('gripper_state passes the device trio through',
                          abs(gs['q'] - 0.25) < 1e-12
                          and abs(gs['force'] - 5.0) < 1e-12)
    no_grip = URBackend(control, receive, None)
    gs = no_grip.gripper_state()
    failures += not check('no gripper -> NaN, so grasp_state says unknown',
                          all(np.isnan(gs[k]) for k in ('q', 'v', 'force')))

    # ---------------- 2F-85 register map: REMOVED ----------------
    #
    # This section drove Robotiq2F85 through a fake Modbus bus. That class and
    # its register map are gone (todo E1, 2026-10-26): connect() builds an RG2,
    # no pymodbus client is ever opened, and nothing in the tree could construct
    # a 2F-85. The checks passed only because the class still existed -- green
    # coverage of a device this cell does not have.

    # ---------------- import graph ----------------
    print('\nIMPORT GRAPH')
    preexisting = 'mujoco' in sys.modules
    import run_hardware  # noqa: F401  (import-time coupling is the test)
    failures += not check(
        'the hardware front-end imports without mujoco',
        (not preexisting) and 'mujoco' not in sys.modules,
        'mujoco in sys.modules: %s' % ('mujoco' in sys.modules))

    print()
    print('=' * 74)
    if failures:
        print('  %d check(s) FAILED' % failures)
        print('=' * 74)
        return 1
    print('  All checks passed. The hardware backend plumbing is sound.')
    print('  This does NOT verify that a real UR3e accepts these calls:')
    print('  run `python tests/test_arm_control.py` against the arm.')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
