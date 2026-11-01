"""Yaw tracking tests, using real cube faces taken from the cell's snapshots.

The cube's top face is extracted from a fixture frame as a polygon, rotated in
the world plane to a known angle, and checked at two levels: that the estimator
follows the rotation, and that the estimate the servo carries follows a full
revolution without stalling or jumping.

Absolute yaw is not known for these frames -- no ground-truth angle was recorded
on the cell -- so the assertions are on the CHANGE in angle, modulo 90.

Run: python tests/test_yaw_sweep.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.perception import _calipers                      # noqa: E402
from src.servo import PBVSServo, _bounded_yaw, _wrap90    # noqa: E402

# Fixture frames, copied here so the test does not depend on the operator's
# snapshot directory.
SHOTS = os.path.join(HERE, 'fixtures', 'shots')

# Cube top face in each fixture: (image patch, depth band in mm). The camera
# looks down, so the face is nearer than the table and forms a plateau inside
# the patch. All three sit at 295-325 mm against a table at ~372 mm.
FACES = {
    'snap_005': ((250, 355, 295, 400), (295, 325)),
    'snap_003': ((250, 355, 295, 400), (295, 325)),
    'snap_001': ((250, 355, 295, 400), (295, 325)),
}

# Rectangle aspect above which the extraction is taken to be wrong. Real faces
# measure 1.02-1.32; a wrong depth band measures ~1.6.
FACE_RATIO_MAX = 1.40

NEAR_M, FAR_M = 0.15, 1.60          # liveview.depth_image's colour map range


def _depth_mm(path):
    # Undo liveview.depth_image: it maps depth to greyscale, then greyscale to
    # TURBO. Render every grey level, match each pixel to its nearest entry, and
    # invert the linear near..far ramp. This is lossy, which is why the
    # assertion below is on a CHANGE of angle and not on an absolute one.
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('cannot read %s' % path)
    greys = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    lut = cv2.applyColorMap(greys, cv2.COLORMAP_TURBO).reshape(-1, 3).astype(np.int32)
    px = img.reshape(-1, 3).astype(np.int32)
    d = ((px[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
    g = d.argmin(axis=1).reshape(img.shape[:2]).astype(float)
    z = (1.0 - g / 255.0) * (FAR_M - NEAR_M) + NEAR_M
    z[g == 0] = 0.0
    return z * 1000.0


def _face_polygon(name):
    # The top face as a polygon in the image plane. Uniform scale is assumed, so
    # pixel coordinates are metric up to a factor and the ANGLE is unaffected.
    from scipy.spatial import ConvexHull
    patch, band = FACES[name]
    z = _depth_mm(os.path.join(SHOTS, name + '.png'))
    y0, y1, x0, x1 = patch
    sub = z[y0:y1, x0:x1]
    ys, xs = np.nonzero((sub > band[0]) & (sub < band[1]))
    if len(xs) < 50:
        raise RuntimeError('%s: only %d face pixels' % (name, len(xs)))
    P = np.column_stack([xs, ys]).astype(float)
    P -= P.mean(axis=0)
    return P[ConvexHull(P).vertices]


def _rot(P, deg):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return P @ R.T


def _track(reads, fold):
    est = None
    out = []
    for r in reads:
        est = fold(est, r)
        out.append(est)
    return np.array(out)


def main():
    failures = 0
    ok = True

    def check(name, cond, detail=''):
        nonlocal failures
        mark = '[PASS]' if cond else '[FAIL]'
        print('  %s %s%s' % (mark, name, ('  -- %s' % detail) if detail else ''))
        if not cond:
            failures += 1

    print('=' * 70)
    print('ARTIFICIAL 360-DEG YAW TEST  (real faces, tests/fixtures/shots/)')
    print('=' * 70)

    # ---- 1. the estimator must follow the rotation exactly ----------------
    print('\n1. ESTIMATOR TRACKS A KNOWN ROTATION')
    for name in FACES:
        H0 = _face_polygon(name)
        yaw0 = _calipers(H0)[1]
        ex = _calipers(H0)[2]
        ratio = max(ex) / max(min(ex), 1e-9)
        print('   %s  hull %2d pts  yaw %6.2f  extents %.1f x %.1f (ratio %.2f)'
              % (name, len(H0), yaw0, ex[0], ex[1], ratio))
        check('%s: the extracted face IS a square (ratio < %.2f)'
              % (name, FACE_RATIO_MAX), ratio < FACE_RATIO_MAX, '%.2f' % ratio)
        errs = []
        for th in range(0, 360, 5):
            y = _calipers(_rot(H0, th))[1]
            errs.append(abs(float(_wrap90(y - (yaw0 + th)))))
        errs = np.array(errs)
        check('%s: 72 rotations, detected yaw follows to within 2 deg' % name,
              errs.max() <= 2.0, 'max %.2f deg' % errs.max())

    # ---- 2. the servo estimate must survive a full revolution ------------
    print('\n2. SERVO ESTIMATE OVER A FULL REVOLUTION  (1 look/degree, +-1.5 deg dither)')
    rng = np.random.default_rng(11)
    truths = np.arange(0.0, 360.0, 1.0)
    reads = truths + rng.normal(0.0, 1.5, truths.shape)

    def fold(est, cand):
        sv = PBVSServo.__new__(PBVSServo)
        sv.yaw_est = est
        sv.yaw_smooth = 0.35
        return PBVSServo._fold_yaw(sv, float(cand))

    ests = _track(reads, fold)
    errs = np.array([abs(float(_wrap90(e - t))) for e, t in zip(ests, truths)])
    steps = np.abs(np.diff(ests))

    check('mean tracking error below 3 deg', errs.mean() < 3.0,
          'mean %.2f' % errs.mean())
    check('never more than 8 deg off over 360 deg', errs.max() < 8.0,
          'max %.2f' % errs.max())
    check('no look is more than 15 deg off (a frozen estimate would fail)',
          (errs > 15.0).sum() == 0, '%d looks' % int((errs > 15.0).sum()))
    check('no single look moves the commanded orientation by more than 10 deg',
          steps.max() < 10.0, 'max %.2f deg' % steps.max())
    check('the estimate actually makes progress round the turn',
          ests[-1] - ests[0] > 300.0, 'moved %.1f deg' % (ests[-1] - ests[0]))

    # ---- 3. the case the old clamp could not represent -------------------
    print('\n3. REGRESSION: A CUBE PARKED ON THE 45-DEG SEAM')
    sv = PBVSServo.__new__(PBVSServo)
    sv.yaw_est = None
    sv.yaw_smooth = 0.35
    for c in (44.5, 45.5, 44.2, 45.8, 44.4, 45.6, 44.9, 45.1):
        sv.yaw_est = PBVSServo._fold_yaw(sv, c)
    check('dithering at 45 settles ON 45, not frozen short of it',
          abs(float(_wrap90(sv.yaw_est - 45.0))) < 1.0,
          'settled %.2f' % sv.yaw_est)

    # ---- 4. the command must stay inside the wrist limit -----------------
    print('\n4. WRIST COMMAND STAYS INSIDE +-60')
    y = 0.0
    cmds = []
    for ref in np.arange(0.0, 1080.0, 1.0):
        y = _bounded_yaw(float(ref), y)
        cmds.append(y)
    cmds = np.array(cmds)
    check('1080 deg of cube rotation: |command| never exceeds 60',
          float(np.max(np.abs(cmds))) <= 60.0 + 1e-9,
          'max %.1f' % np.max(np.abs(cmds)))

    # A cube sitting at the boundary with measurement noise must latch to ONE
    # exact branch, not alternate between +60 and the opposite face.
    rng2 = np.random.default_rng(21)
    last = 60.0
    noisy = []
    refs = []
    for _ in range(100):
        ref = float(60.0 + rng2.normal(0.0, 0.5))
        last = _bounded_yaw(ref, last)
        refs.append(ref)
        noisy.append(last)
    noisy = np.array(noisy)
    flips = int(np.sum(np.abs(np.diff(np.sign(noisy))) > 0))
    check('yaw noise at +60 latches one branch (at most one switch)',
          flips <= 1, '%d sign changes' % flips)
    check('every noise command is within 2 deg of an exact face and in limit',
          max(abs(_wrap90(c - r)) for c, r in zip(noisy, refs)) <= 2.0 + 1e-9
          and np.max(np.abs(noisy)) <= 60.0 + 1e-9,
          'max face error %.2f deg'
          % max(abs(_wrap90(c - r)) for c, r in zip(noisy, refs)))

    c66 = _bounded_yaw(66.0, 60.0)
    c100 = _bounded_yaw(100.0, 60.0)
    check('past the limit, the command is an exact face inside the limit',
          abs(_wrap90(c66 - 66.0)) < 1e-9
          and abs(_wrap90(c100 - 100.0)) < 1e-9
          and abs(c66) <= 60.0 + 1e-9 and abs(c100) <= 60.0 + 1e-9,
          '66 -> %.1f, 100 -> %.1f' % (c66, c100))
    check('a wide accumulated yaw still returns inside the limit',
          abs(_bounded_yaw(720.0, 720.0)) <= 60.0 + 1e-9,
          'command %.1f' % _bounded_yaw(720.0, 720.0))

    print()
    if failures:
        print('%d FAILED' % failures)
        return 1
    print('all yaw sweep checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
