# Layering regression test: the robot must not depend on the world's guts.
#
# WHY THIS EXISTS
# ---------------
# The robot side of this project has to run on a real UR3e. The object side is
# a simulation demonstration (a cube glided by writing qvel, teleported into
# place, scored by reading its true pose). Mixing the two is what makes a
# simulator-only prototype unportable, and it is the kind of coupling that
# creeps back in one convenient `import` at a time.
#
# So the rule is enforced mechanically rather than by comment:
#
#   ROBOT modules may import only: config, overlay, perception (geometry),
#   tracking, controller, observer, state_space, world.
#
#   ROBOT modules must NOT import: target_motion, pick_place, model_loader,
#   or camera/calibration (the sensor and scene are the world's business).
#
# World-facing modules (main, run_tests, demo_servo, pick_place, world,
# calibration, model_loader, overlay, target_motion) may import anything.
#
# Run:  python tests/test_layering.py      (or `python -m pytest tests/`)

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Modules that must stay portable to hardware.
ROBOT_MODULES = [
    'arm.py',
    'servo.py',
    'controller.py',
    'observer.py',
    'state_space.py',
    'tracking.py',
    'task.py',
]

# Modules the robot may not reach for, and why.
FORBIDDEN = {
    'target_motion': 'the cube driver is a simulation-only demonstration',
    'pick_place': 'scene layout, spawning and object pool bookkeeping',
    'model_loader': 'scene composition (table, chute, object pool)',
    'camera': 'the sensor belongs behind the world boundary',
    'calibration': 'an offline camera rig, not a runtime dependency',
    'overlay': 'the viewer overlay layer; robot code pushes via duck typing',
}

# Third-party imports that only exist in simulation.
FORBIDDEN_TOPLEVEL = {
    'mujoco': 'the physics engine is simulation-only',
    'mujoco.viewer': 'there is no viewer on hardware',
    'matplotlib': 'plotting belongs to the harness',
}

# Calls that would reach object state even without a forbidden import.
FORBIDDEN_CALLS = [
    ('reg.pose', 'object ground truth'),
    ('reg.teleport', 'object mutation'),
    ('reg.freeze', 'object mutation'),
    ('reg.qpos_adrs', 'raw freejoint offsets'),
    ('reg.qvel_adrs', 'raw freejoint offsets'),
    ('ObjectRegistry', 'the object pool'),
    ('TargetMover', 'the cube driver'),
    ('spawn_one', 'object spawning'),
    ('place_cube', 'object spawning'),
    ('Perceive_all(', 'bypassing the world boundary'),
]


def _imported_names(tree, path):
    # Top-level and function-level imports alike: a deferred import inside a
    # function body still couples the modules, it just hides the cycle.
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod.startswith('src.'):
                found.append(mod[len('src.'):].split('.')[0])
            elif mod:
                found.append(mod.split('.')[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith('src.'):
                    found.append(a.name[len('src.'):].split('.')[0])
                else:
                    found.append(a.name.split('.')[0])
    return found


def check_imports():
    failures = []
    for name in ROBOT_MODULES:
        path = os.path.join(SRC, name)
        with open(path) as fh:
            src = fh.read()
        tree = ast.parse(src, path)
        for mod in set(_imported_names(tree, path)):
            if mod in FORBIDDEN:
                failures.append(
                    '  %s imports %s  -- %s' % (name, mod, FORBIDDEN[mod]))
            # mujoco is needed by the servo/observer, so only task.py is held
            # to the no-physics-engine rule.
            if name == 'task.py' and mod in FORBIDDEN_TOPLEVEL:
                failures.append(
                    '  %s imports %s  -- %s'
                    % (name, mod, FORBIDDEN_TOPLEVEL[mod]))
    return failures


def check_calls():
    # Belt and braces: a robot module could reach object state through a
    # parameter rather than an import (e.g. being handed `reg`). Look for the
    # accessor names in the source, ignoring comments.
    failures = []
    for name in ROBOT_MODULES:
        path = os.path.join(SRC, name)
        with open(path) as fh:
            lines = fh.readlines()
        for i, raw in enumerate(lines, 1):
            code = raw.split('#', 1)[0]
            for needle, why in FORBIDDEN_CALLS:
                if needle in code:
                    failures.append('  %s:%d reaches %s (%s)'
                                    % (name, i, needle, why))
    return failures


def check_hardware_signatures():
    # Arm must not require a viewer: the trajectory executor has to run with no
    # window. `viewer=` may only appear as an optional compatibility alias.
    path = os.path.join(SRC, 'arm.py')
    with open(path) as fh:
        src = fh.read()
    failures = []
    if 'from src.overlay import' in src:
        failures.append('  arm.py imports the overlay module directly; it must '
                        'use duck typing so it compiles with no viewer')
    if 'self.mover' in src:
        failures.append('  arm.py still owns a mover; the world must own it')
    return failures


def check_world_tick_rate():
    # A physics loop that calls mj_step must also tick the world, or the
    # object's demonstration motion runs at the wrong rate. This bit us once:
    # driving the mover from viewer.sync moved the cube at 1/SYNC_STEPS of its
    # true speed, because sync runs every 8 physics steps.
    #
    # Heuristic and deliberately conservative: in the robot modules and the
    # task layer, every `mujoco.mj_step` in a live code path should have a
    # `step_mover`/`_step_world` call nearby.
    failures = []
    for name in ('servo.py', 'pick_place.py'):
        path = os.path.join(SRC, name)
        with open(path) as fh:
            lines = fh.readlines()
        for i, raw in enumerate(lines, 1):
            if 'mujoco.mj_step' not in raw.split('#', 1)[0]:
                continue
            window = ''.join(
                l.split('#', 1)[0] for l in lines[max(0, i - 8):i + 8])
            if ('step_mover' not in window and '_step_world' not in window):
                failures.append(
                    '  %s:%d calls mj_step with no world tick nearby; the '
                    'object demo will run at the wrong rate' % (name, i))
    return failures


def main():
    groups = [
        ('robot -> world imports', check_imports()),
        ('robot -> object state access', check_calls()),
        ('hardware signature constraints', check_hardware_signatures()),
        ('world tick rate vs physics steps', check_world_tick_rate()),
    ]

    print()
    print('=' * 74)
    print('LAYERING CHECK  (robot must not depend on the world)')
    print('=' * 74)
    bad = 0
    for title, failures in groups:
        if failures:
            bad += len(failures)
            print('FAIL  %s' % title)
            for f in failures:
                print(f)
        else:
            print('ok    %s' % title)
    print()

    if bad:
        print('%d violation(s): the robot side is not hardware-portable.' % bad)
        print('Move the offending logic behind src.world.ObjectWorld, or into')
        print('src/pick_place.py / src/demo_servo.py if it is scene or scoring.')
        return 1

    print('Robot modules: %s' % ', '.join(m[:-3] for m in ROBOT_MODULES))
    print('All clean -- robot side depends only on the world boundary.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
