"""Task-flow test: the pickup key is a toggle, and a grasp does not release.

WHY THIS EXISTS
---------------
The operator-facing rule is: press the pickup key and the robot TAKES the
cube and holds it; press again and it sets it down. That rule used to be
violated inside the pick routine, which grasped, carried and released in one
call -- so the second press had nothing to do.

The fix split the release out of the pick (src/pick.py: pick_and_hold) and
made the state machine own the toggle. This test pins that behaviour with
stubs, so it runs without MuJoCo and without a robot.

Run: python tests/test_task_flow.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.task import Commands, TaskStateMachine, WAIT, TRACK, HOLD, PUTDOWN


class StubWorld:
    def __init__(self):
        self.present = True

    def active(self):
        return [0] if self.present else []


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))
    return bool(ok)


def main():
    failures = 0
    log = []
    calls = {'pick': 0, 'put': 0, 'home': 0}

    world = StubWorld()
    cmd = Commands()

    def pick_object():
        calls['pick'] += 1
        return True

    def put_object():
        calls['put'] += 1
        return True

    def go_home():
        calls['home'] += 1

    task = TaskStateMachine(
        commands=cmd, say=lambda m: log.append(m), at_home=lambda: True,
        go_home=go_home, place_object=lambda: None,
        track_step=lambda: None, pick_object=pick_object,
        running=lambda: True, arm=None, world=world)
    task.on_put_object = put_object

    print()
    print('=' * 70)
    print('PICKUP TOGGLE  (hold until the key is pressed again)')
    print('=' * 70)
    print('  initial mode: %s' % task.mode)
    failures += not check('starts in WAIT', task.mode == WAIT)

    # first press: grasp, then hold -- and NOT release
    cmd.pickup = True
    task.step()
    failures += not check('one press runs the pick', calls['pick'] == 1,
                          'pick=%d' % calls['pick'])
    failures += not check('the robot is holding afterwards', task.holding)
    failures += not check('the mode is HOLD, not PUTDOWN', task.mode == HOLD,
                          'mode=%s' % task.mode)
    failures += not check('the cube was NOT released by the pick',
                          calls['put'] == 0, 'put=%d' % calls['put'])

    # stepping while held must not release anything
    for _ in range(5):
        task.step()
    failures += not check('waiting in HOLD does not release',
                          calls['put'] == 0 and task.mode == HOLD)

    # second press: put down, once
    cmd.pickup = True
    task.step()
    failures += not check('the second press releases exactly once',
                          calls['put'] == 1, 'put=%d' % calls['put'])
    failures += not check('holding is cleared', not task.holding)
    failures += not check('the arm retracts to home after the release',
                          calls['home'] == 1, 'home=%d' % calls['home'])
    failures += not check('the mode returns to TRACK', task.mode == TRACK,
                          'mode=%s' % task.mode)
    failures += not check('PUTDOWN was reported',
                          any('PUTDOWN' in m for m in log))

    # a third press grasps again
    cmd.pickup = True
    task.step()
    failures += not check('the third press grasps again', calls['pick'] == 2,
                          'pick=%d' % calls['pick'])
    failures += not check('and holds again', task.holding and task.mode == HOLD)

    print()
    print('=' * 70)
    if failures:
        print('  %d check(s) FAILED' % failures)
        print('=' * 70)
        return 1
    print('  Pickup toggles grasp -> hold -> put down -> grasp.')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
