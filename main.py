#!/usr/bin/env python3
# Entry point. Dispatches to a front-end.
#
#   python main.py              -> simulation (current default)
#   python main.py --sim        -> simulation, explicit
#   python main.py --hardware   -> real robot (run_hardware.py)
#
# This file is deliberately minimal: it decides WHICH front-end runs and does
# nothing else. All simulation concerns live in run_simulation.py; all hardware
# concerns in run_hardware.py; all portable task logic lives in src/task.py.
#
# NOTE ON THE DEFAULT
# -------------------
# The default is simulation for now so the project is runnable as-is. On the
# cell the sensible default becomes the real robot and simulation should
# require --sim. That is a one-line change in `_parse` below.

import argparse
import sys


def _parse(argv=None):
    ap = argparse.ArgumentParser(
        prog='main.py',
        description='Vision-guided pick: choose a front-end.',
        epilog='Simulation is the default; --hardware needs the cell.')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sim', '--simulation', dest='sim', action='store_true',
                   help='run the MuJoCo simulation (default)')
    g.add_argument('--hardware', '--real', dest='hardware', action='store_true',
                   help='run on the real robot (see run_hardware.py)')
    # Anything else belongs to the chosen front-end (e.g. --ip, --no-gripper),
    # so it is collected here rather than rejected.
    args, rest = ap.parse_known_args(argv)
    args.rest = rest
    return args


def run_simulation():
    # Imported lazily so that --hardware does not pull in mujoco/the viewer.
    from run_simulation import main as sim_main
    return sim_main()


def run_hardware(args):
    # Imported lazily so --sim never pulls in ur_rtde/pymodbus, and --hardware
    # never pulls in mujoco.
    from run_hardware import main as hw_main
    return hw_main(getattr(args, 'rest', []))


def main(argv=None):
    args = _parse(argv)
    if args.hardware:
        return run_hardware(args)
    # Default: simulation.
    return run_simulation()


if __name__ == '__main__':
    sys.exit(main())
