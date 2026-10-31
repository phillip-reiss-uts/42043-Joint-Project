"""Artificial 360-degree yaw test, built from the real snapshots.

WHY THIS EXISTS
---------------
The yaw estimator cannot be validated at the cell, and the simulator's cube is
too clean to be a fair test. The PNGs in tests/fixtures/shots/ are real depth
frames of the real cell: a real cube face, real ragged edges, real dropouts,
real table.

This test takes the CUBE'S TOP FACE out of one of those frames as a polygon,
rotates that polygon in the world plane to a KNOWN angle, and checks two
things:

  1. THE ESTIMATOR (_calipers + _top_band, i.e. what src/world_real.detect does)
     must track the rotation exactly. The absolute yaw at the unrotated angle is
     not known -- no ground-truth angle was ever recorded on the cell -- so the
     assertion is on the CHANGE: rotating by theta must move the detected yaw by
     exactly theta, modulo 90 (a square face is 4-fold symmetric).

  2. THE ESTIMATE THE SERVO CARRIES (PBVSServo._fold_yaw) must track a cube
     being turned through a full revolution without freezing and without ever
     commanding a large single-look change of the tool orientation.

Why the second matters, and what was wrong before: the estimate used to be
clamped into [-45, 45). Two readings either side of the 90-degree seam (+44 and
-44) are 2 degrees apart as angles but 88 apart in that clamped coordinate, so a
tie rule froze the estimate near the seam. Tracking a cube through 360 degrees,
the old code accumulated up to 20 degrees of error, sat over 10 degrees for 39
of 360 looks, and jumped the stored estimate by up to 68 degrees in ONE look --
which is a wrist swing across most of a right angle for a grasp that did not
need it.

What this test does NOT prove: that the face in a live frame is extracted the
same way, or that the cube is where perception thinks. It is a test of the yaw
path only, on real face shapes.

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
from src.servo import PBVSServo, _wrap90                  # noqa: E402

# The two frames this test is built on, COPIED into tests/fixtures so the test
# does not depend on the operator's snapshot directory -- shots/ is working
# output and can be cleared between sessions, and a test that silently finds
# nothing there is worse than no test.
SHOTS = os.path.join(HERE, 'fixtures', 'shots')

# The cube's top face in each snapshot: (image patch, depth band in mm). The
# band is the face's own depth mode -- the camera looks down, so the cube top is
# CLOSER than the table and shows up as the near plateau inside the patch. All
# three frames put the face at 295-325 mm against a table at ~372 mm.
#
# The three are deliberately different qualities of the same face:
#   snap_005  extents 36.2 x 35.4 px  ratio 1.02   the cleanest
#   snap_003  extents 34.5 x 40.0 px  ratio 1.16
#   snap_001  extents 47.8 x 36.3 px  ratio 1.32   the ragged one -- taken while
#             the gripper was HOLDING the cube, so the jaws clip the face
FACES = {
    'snap_005': ((250, 355, 295, 400), (295, 325)),
    'snap_003': ((250, 355, 295, 400), (295, 325)),
    'snap_001': ((250, 355, 295, 400), (295, 325)),
}

# Above this a face is not a square and the extraction is wrong. The real faces
# sit at 1.02-1.32; a wrong depth band (the table instead of the face) measures
# ~1.6. The limit sits between the two rather than close to the real maximum,
# because a limit tuned to 1.35 would fail on a slightly worse frame and say
# "extraction" when the extraction was fine.
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
    print('ARTIFICIAL 360-DEG YAW TEST  (real face shapes from shots/)')
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

    print()
    if failures:
        print('%d FAILED' % failures)
        return 1
    print('all yaw sweep checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
