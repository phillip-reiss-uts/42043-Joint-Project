"""Headless batch harness: run N pick-and-place attempts and report a rate.

Usage:
    python -m src.batch                 # 20 attempts, default seed
    python -m src.batch --n 50 --seed 3 --verbose

Runs without a viewer. Builds the model, camera and arm ONCE -- RGBDCamera owns
a GPU renderer context, and creating one per attempt exhausts the context after
a handful of iterations.
"""

import argparse
import time
from collections import Counter

import mujoco
import numpy as np

from src.arm import Arm
from src.camera import RGBDCamera
from src.config import TABLE_HEIGHT, OBJECT_HEIGHT
from src.model_loader import load_robot_model
from src.perception import find_camera
from src.pick_place import pick_and_place


def run_batch(n=20, seed=0, verbose=False):
    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    cam_name, cam_id = find_camera(model)
    if cam_id < 0:
        raise SystemExit('no wrist camera found')

    camera = RGBDCamera(model, cam_name, width=640, height=480)
    arm = Arm(model, data)
    rng = np.random.default_rng(seed)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    rest_z = TABLE_HEIGHT + OBJECT_HEIGHT / 2.0

    counts = Counter()
    perrs, place_errs = [], []

    print(f'[BATCH] {n} attempts, seed {seed}')
    print(f'{"#":>4}  {"outcome":<14} {"perr mm":>8} {"place mm":>9} '
          f'{"yaw":>6} {"smp":>4} {"pk":>3}  object end')

    t0 = time.time()
    for i in range(n):
        result = pick_and_place(model, data, arm, camera, cam_id, rng,
                                viewer=None, aborted=None)

        end = np.array(data.xpos[body_id]).copy()
        on_table = abs(end[2] - rest_z) < 0.005

        if result.aborted:
            outcome = 'aborted'
        elif not result.found:
            outcome = 'no-perception'
        elif result.picked and result.placed:
            outcome = 'placed'
        elif result.picked:
            outcome = 'dropped'
        else:
            outcome = 'pick-failed'
        counts[outcome] += 1

        if not np.isnan(result.perr):
            perrs.append(result.perr)
        if not np.isnan(result.placed_err):
            place_errs.append(result.placed_err)

        print(f'{i:>4}  {outcome:<14} '
              f'{result.perr * 1000:>8.1f} {result.placed_err * 1000:>9.1f} '
              f'{result.yaw:>6.1f} {result.n_samples:>4} {result.n_peeks:>3}  '
              f'({end[0]:+.3f},{end[1]:+.3f},{end[2]:+.3f})'
              f'{"  ON TABLE" if on_table else ""}')

    dt = time.time() - t0
    print()
    print(f'[BATCH] results over {n} attempts in {dt:.0f} s '
          f'({dt / max(n, 1):.1f} s/attempt)')
    for k in ('placed', 'dropped', 'pick-failed', 'no-perception', 'aborted'):
        if counts[k]:
            print(f'  {k:<14} {counts[k]:>4}  ({100 * counts[k] / n:.0f}%)')
    if perrs:
        print(f'  perception err mean {np.mean(perrs) * 1000:.2f} mm, '
              f'max {np.max(perrs) * 1000:.2f} mm')
    if place_errs:
        print(f'  placement err  mean {np.mean(place_errs) * 1000:.2f} mm, '
              f'max {np.max(place_errs) * 1000:.2f} mm')

    return counts


def main():
    ap = argparse.ArgumentParser(description='headless pick-and-place batch')
    ap.add_argument('--n', type=int, default=20, help='attempts (default 20)')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed (default 0)')
    ap.add_argument('--verbose', action='store_true',
                    help='print each stage; implies config VERBOSE')
    args = ap.parse_args()

    if args.verbose:
        import src.config as cfg
        cfg.VERBOSE = True

    run_batch(n=args.n, seed=args.seed)


if __name__ == '__main__':
    main()