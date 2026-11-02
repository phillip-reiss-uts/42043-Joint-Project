"""Layering regression test: the robot must not depend on the world's guts.

WHY THIS EXISTS
---------------
The robot side of this project has to run on a real UR3e. The object side is
a simulation demonstration (a cube glided by writing qvel, teleported into
place, scored by reading its true pose). Mixing the two is what makes a
simulator-only prototype unportable, and it is the kind of coupling that
creeps back in one convenient `import` at a time.

So the rule is enforced mechanically rather than by comment:

  ROBOT modules may import only: config, overlay, perception (geometry),
  tracking, controller, observer, state_space, world.

  ROBOT modules must NOT import: target_motion, pick_place, model_loader,
  or camera/calibration (the sensor and scene are the world's business).

World-facing modules (main, run_tests, pick_place, world_sim, calibration,
model_loader, overlay, target_motion) may import anything.

PROVIDER SEAM (second rule, no simulator behind the interfaces)
--------------------------------------------------------------
The three seams that let the same robot code drive MuJoCo or a real UR3e are:

    src/kinematics.py       Kinematics      -> src/kinematics_mujoco.py
    src/robot_backend.py    RobotBackend    -> src/robot_backend_mujoco.py
    src/world.py            ObjectWorld     -> src/world_sim.py

The left-hand modules are SHARED: they must not import mujoco, and must not
import any of the right-hand implementation modules. The implementations are
simulation-only and are chosen by the entry point, never by the robot layer.

Run:  python tests/test_layering.py      (or `python -m pytest tests/`)
"""

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
    'pick.py',
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

# Provider seam: interface module -> (implementation module, required class).
PROVIDER_SEAM = {
    'kinematics.py': ('kinematics_mujoco.py', 'MujocoKinematics',
                      'Kinematics'),
    'robot_backend.py': ('robot_backend_mujoco.py', 'MujocoBackend',
                         'RobotBackend'),
    'world.py': ('world_sim.py', 'SimObjectWorld', 'ObjectWorld'),
}

# Implementation modules that are simulation-only.
SIM_IMPLEMENTATIONS = ('kinematics_mujoco', 'robot_backend_mujoco', 'world_sim')

# Implementation modules that are hardware-only. A robot module must not import
# these either: the front-end chooses the provider and injects it.
HARDWARE_IMPLEMENTATIONS = ('kinematics_ur', 'robot_backend_ur')

# Known, deliberate legacy couplings. These two modules still build a MuJoCo
# provider when the caller passes no `kin=`, so that the demos and older call
# styles keep working without threading a provider through every signature.
# They are allow-listed rather than ignored so the deviation is visible and
# cannot spread to another robot module.
LEGACY_KIN_DEFAULTS = ('observer.py', 'state_space.py')


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
            # task.py, arm.py and servo.py are fully mujoco-free: geometry comes
            # from src/kinematics.py and the clock from src/robot_backend.py.
            if name in ('task.py', 'arm.py', 'servo.py') and mod in FORBIDDEN_TOPLEVEL:
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


def check_kinematics_seam():
    # FK, the Jacobian, M(q) and g(q) must go through the Kinematics provider
    # (src/kinematics.py), not through direct MuJoCo calls in the control layer.
    # This is the seam that lets the same control logic drive a real UR3e.
    #
    # `mj_step` is exempt: it is the physics-clock question the driver layer
    # owns, not a kinematics query.
    KIN_CALLS = ('mj_kinematics', 'mj_comPos', 'mj_jacBody', 'mj_fullM',
                 'mj_crb', 'mj_factorM', 'mj_rne')
    failures = []
    for name in ROBOT_MODULES:
        path = os.path.join(SRC, name)
        with open(path) as fh:
            lines = fh.readlines()
        for i, raw in enumerate(lines, 1):
            code = raw.split('#', 1)[0]
            for call in KIN_CALLS:
                if call in code:
                    failures.append(
                        '  %s:%d calls %s directly; use a Kinematics provider'
                        % (name, i, call))
    return failures


def check_world_tick_rate():
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


def check_no_mujoco_imports():
    # The tightest robot modules must import no simulator at all: their geometry
    # comes from src/kinematics.py and their clock from src/robot_backend.py.
    #
    # Only TOP-LEVEL imports are failures. A function-body import is still a
    # coupling, but it is the deliberate shape of the legacy compatibility
    # shims (e.g. state_space._default_kin), where the provider is built lazily
    # so the module has no simulator dependency at import time.
    failures = []
    for name in ('task.py', 'arm.py', 'servo.py', 'pick.py', 'controller.py',
                 'tracking.py', 'state_space.py', 'kinematics.py',
                 'robot_backend.py', 'arm_dynamics.py'):
        path = os.path.join(SRC, name)
        with open(path) as fh:
            tree = ast.parse(fh.read(), path)
        for node in tree.body:            # module level only
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split('.')[0] in ('mujoco',):
                        failures.append(
                            '  %s:%d imports %s at module level; use '
                            'Kinematics/RobotBackend'
                            % (name, node.lineno, a.name))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or '').split('.')[0] == 'mujoco':
                    failures.append(
                        '  %s:%d imports from %s at module level'
                        % (name, node.lineno, node.module))
    return failures


def _module_imports(tree):
    # Every import in the file, including function bodies. A deferred import
    # still couples two modules; it only hides the cycle.
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod:
                found.add(mod.split('.')[0])
                if mod.startswith('src.'):
                    found.add(mod[len('src.'):].split('.')[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split('.')[0])
                if a.name.startswith('src.'):
                    found.add(a.name[len('src.'):].split('.')[0])
    return found


def check_provider_seam():
    # The three interfaces must be implementation-free, and the implementations
    # must actually implement them. This is what makes "shared code" a fact
    # rather than a comment: nothing in src/kinematics.py, src/robot_backend.py
    # or src/world.py can reach a simulator.
    failures = []
    for iface, (impl, cls, base) in PROVIDER_SEAM.items():
        ipath = os.path.join(SRC, iface)
        with open(ipath) as fh:
            tree = ast.parse(fh.read(), ipath)
        imports = _module_imports(tree)
        if 'mujoco' in imports:
            failures.append('  %s imports mujoco; it is the shared interface'
                            % iface)
        for sim in SIM_IMPLEMENTATIONS:
            if sim in imports:
                failures.append('  %s imports %s; the implementation is chosen '
                                'by the entry point, not the interface'
                                % (iface, sim))

        ppath = os.path.join(SRC, impl)
        if not os.path.isfile(ppath):
            failures.append('  %s is missing' % impl)
            continue
        with open(ppath) as fh:
            ptree = ast.parse(fh.read(), ppath)
        classes = {n.name: n for n in ast.walk(ptree)
                   if isinstance(n, ast.ClassDef)}
        if cls not in classes:
            failures.append('  %s does not define %s' % (impl, cls))
        elif base not in [getattr(b, 'id', None) for b in classes[cls].bases]:
            failures.append('  %s.%s does not subclass %s'
                            % (impl, cls, base))

    # No robot module outside the documented legacy pair may reach for a
    # simulation implementation.
    for name in ROBOT_MODULES:
        if name in LEGACY_KIN_DEFAULTS:
            continue
        path = os.path.join(SRC, name)
        with open(path) as fh:
            tree = ast.parse(fh.read(), path)
        for mod in _module_imports(tree):
            if mod in SIM_IMPLEMENTATIONS or mod in HARDWARE_IMPLEMENTATIONS:
                failures.append('  %s imports %s; inject the provider instead'
                                % (name, mod))
    return failures


def main():
    groups = [
        ('robot -> world imports', check_imports()),
        ('robot -> object state access', check_calls()),
        ('hardware signature constraints', check_hardware_signatures()),
        ('world tick rate vs physics steps', check_world_tick_rate()),
        ('kinematics seam (no direct MuJoCo geometry)', check_kinematics_seam()),
        ('no mujoco imports in the tight robot modules',
         check_no_mujoco_imports()),
        ('provider seam (interfaces stay implementation-free)',
         check_provider_seam()),
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
        print('src/pick_place.py if it is scene or scoring.')
        return 1

    print('Robot modules: %s' % ', '.join(m[:-3] for m in ROBOT_MODULES))
    print('All clean -- robot side depends only on the world boundary.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
