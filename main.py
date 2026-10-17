#!/usr/bin/env python3
# Entry point. Dispatches to a front-end.
#
#   python main.py              -> simulation (current default)
#   python main.py --sim        -> simulation, explicit
#   python main.py --hardware   -> real robot (NOT IMPLEMENTED YET)
#
# This file is deliberately minimal: it decides WHICH front-end runs and does
# nothing else. All simulation concerns live in run_simulation.py; all portable
# task logic lives in src/task.py.
#
# NOTE ON THE DEFAULT
# -------------------
# The default is simulation for now so the project is runnable as-is. Once the
# hardware front-end exists, the sensible default becomes the real robot and
# simulation should require --sim. That is a one-line change in `_parse` below.

import argparse
import sys


def _parse(argv=None):
    ap = argparse.ArgumentParser(
        prog='main.py',
        description='Vision-guided pick: choose a front-end.',
        epilog='Simulation is the default until the hardware front-end exists.')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sim', '--simulation', dest='sim', action='store_true',
                   help='run the MuJoCo simulation (default)')
    g.add_argument('--hardware', '--real', dest='hardware', action='store_true',
                   help='run on the real robot (not implemented yet)')
    return ap.parse_args(argv)


def run_simulation():
    # Imported lazily so that --hardware does not pull in mujoco/the viewer.
    from run_simulation import main as sim_main
    return sim_main()


def run_hardware():
    print('[ERROR] The hardware front-end is not implemented yet.')
    print()
    print('The robot side is already portable: src/task.py, src/arm.py,')
    print('src/servo.py and src/controller.py contain no mujoco or viewer.')
    print('What is missing is a second ObjectWorld (see src/world.py) and')
    print('real drivers:')
    print('  * RealObjectWorld backed by RealSense captures')
    print('  * UR3e forward kinematics for gripper_pose()')
    print('  * UR RTDE/URScript joint moves for Arm.move')
    print('  * Modbus reads for the 2F-85 registers Arm.grasp_state uses')
    return 2


def main(argv=None):
    args = _parse(argv)
    if args.hardware:
        return run_hardware()
    # Default: simulation.
    return run_simulation()


if __name__ == '__main__':
    sys.exit(main())
