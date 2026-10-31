#!/usr/bin/env python3
# Entry point. Dispatches to a front-end.
#
#   python main.py              -> simulation (current default)
#   python main.py --sim        -> simulation, explicit
#   python main.py --hardware   -> real robot (run_hardware.py)
#
#   python main.py --lightweight   simulation with no camera panels; for older
#                                  machines. See the flag's help.
#
# This file is deliberately minimal: it decides WHICH front-end runs and does
# nothing else. All simulation concerns live in run_simulation.py; all hardware
# concerns in run_hardware.py; all portable task logic lives in src/task.py.
#
# When called without arguments, this file will run the simulation. On the
# cell call --hardware or --real. Simulation can also be run by calling --sim.

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
    ap.add_argument('--lightweight', action='store_true',
                    help='simulation only: drop the camera image and '
                         'point-cloud panels. Each is its own GL render per '
                         'frame, inteded for older hardware, or resource constrained environments. it '
                         'The 3D scene, trail, detections and text panels all stay.')
    # Anything else belongs to the chosen front-end (e.g. --ip, --no-gripper),
    # so it is collected here rather than rejected.
    args, rest = ap.parse_known_args(argv)
    args.rest = rest
    return args


def run_simulation(args=None):
    # Imported lazily so that --hardware does not pull in mujoco/the viewer.
    from run_simulation import main as sim_main
    return sim_main(lightweight=bool(getattr(args, 'lightweight', False)))


def run_hardware(args):
    # Imported lazily so --sim never pulls in ur_rtde, and --hardware never
    # pulls in mujoco.
    from run_hardware import main as hw_main
    return hw_main(getattr(args, 'rest', []))


def main(argv=None):
    args = _parse(argv)
    if args.hardware:
        return run_hardware(args)
    # Default: simulation.
    return run_simulation(args)


if __name__ == '__main__':
    sys.exit(main())
