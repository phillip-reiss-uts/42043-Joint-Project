# Fixed-camera (eye-to-hand) tests: offline, no robot, no RealSense.
#
# WHAT THIS PROVES
# ----------------
# `FixedCameraWorld` is the eye-to-hand counterpart of `RealObjectWorld`: the
# camera sits on the cell frame, so its pose is a constant in the base frame
# rather than FK(q) @ mount. Two properties matter and are easy to get wrong:
#
#   1. the constant pose must be in the same convention the deprojection uses
#      (the MuJoCo camera one: columns = right, up, backward);
#   2. moving the ARM must not move the detection -- that is the whole point of
#      a fixed camera, and the bug would be silent, because the robot would
#      simply chase an estimate that drifts with its own pose.
#
# `FakeDepthSource` renders a fixed, top-down camera, which is exactly the
# geometry a fixed cell camera has, so the whole path is testable here.
#
# Run: python tests/test_fixed_camera.py

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.world import ObjectWorld
from src.world_real import (
    FakeDepthSource, FixedCameraWorld, RealObjectWorld, camera_transform,
)

CAM_HEIGHT = 0.62          # the fake source's table distance
# The fake intrinsics put the principal point 0.6 px off the pixel centre and
# the cube is a quantised 18-pixel patch, so a few mm of offset is expected at
# 0.62 m. A wrong convention would be centimetres, not millimetres.
TOL_MM = 6.0


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))
    return bool(ok)


def main():
    failures = 0
    print()
    print('=' * 74)
    print('FIXED CAMERA  (eye-to-hand, offline)')
    print('=' * 74)

    # ---------------- transform convention ----------------
    T = camera_transform((0.0, 0.0, CAM_HEIGHT), (0.0, 0.0, -1.0),
                         (0.0, 1.0, 0.0))
    R = T[:3, :3]
    p = T[:3, 3]
    print('\nTRANSFORM')
    failures += not check(
        'the rotation is orthonormal with det +1',
        np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        and abs(np.linalg.det(R) - 1.0) < 1e-12)
    failures += not check(
        'a camera looking down maps to the identity rotation',
        np.allclose(R, np.eye(3), atol=1e-12),
        'R diag %s' % np.round(np.diag(R), 3))
    failures += not check(
        'the position is carried through',
        np.allclose(p, [0.0, 0.0, CAM_HEIGHT]))
    failures += not check(
        'the third column points backwards along the view',
        np.allclose(R[:, 2], [0.0, 0.0, 1.0]))
    Tt = camera_transform((0.3, -0.2, 0.5), (0.3, -0.1, 0.0))
    failures += not check(
        'an oblique aim still yields a valid rotation',
        np.allclose(Tt[:3, :3] @ Tt[:3, :3].T, np.eye(3), atol=1e-12)
        and abs(np.linalg.det(Tt[:3, :3]) - 1.0) < 1e-12)

    # ---------------- detection ----------------
    src = FakeDepthSource(320, 240)
    world = FixedCameraWorld(src, lambda: (np.zeros(3), np.eye(3)), T)
    print('\nDETECTION')
    failures += not check('is an ObjectWorld', isinstance(world, ObjectWorld))
    views = world.perceive()
    failures += not check('finds the cube the fake camera rendered',
                          len(views) == 1, 'views=%d' % len(views))
    if views:
        c = views[0].centre
        err = float(np.linalg.norm(c[:2]))
        failures += not check('detects it on the optical axis',
                              err * 1e3 < TOL_MM,
                              'centre xy %s mm from the axis'
                              % np.round(c[:2] * 1e3, 2))

    # ---------------- pose independence ----------------
    # The arm moves; the camera does not. A fixed-camera world must ignore the
    # flange completely for perception (it still reports it for gripper_pose).
    poses = [(np.zeros(3), np.eye(3)),
             (np.array([0.3, 0.2, 0.4]), np.eye(3)),
             (np.array([0.1, -0.1, 0.9]), np.eye(3))]
    seen = [np.array(world.perceive()[0].centre) for _ in poses]
    world._flange_pose = lambda: poses[0]   # noqa: SLF001 (test hook)
    for pose in poses[1:]:
        world._flange_pose = (lambda p: (lambda: p))(pose)
        seen.append(np.array(world.perceive()[0].centre))
    spread = float(np.max(np.linalg.norm(np.array(seen) - seen[0], axis=1)))
    print('\nEYE-TO-HAND')
    failures += not check(
        'moving the arm does not move the estimate',
        spread < 1e-12, 'max shift %.2e m over three arm poses' % spread)

    # ---------------- equivalence with the wrist path ----------------
    # If the arm happens to hold a wrist camera exactly where the fixed one is,
    # the two worlds must agree to the bit.
    src2 = FakeDepthSource(320, 240)
    held = (np.array([0.0, 0.0, CAM_HEIGHT]), np.eye(3))
    wrist = RealObjectWorld(src2, lambda: held, np.eye(4))
    v_wrist = wrist.perceive()
    v_fixed = FixedCameraWorld(FakeDepthSource(320, 240),
                               lambda: held, T).perceive()
    print('\nCONSISTENCY')
    failures += not check(
        'the fixed world matches the wrist world at the same camera pose',
        len(v_wrist) == len(v_fixed) == 1
        and np.allclose(v_wrist[0].centre, v_fixed[0].centre, atol=1e-12),
        'centres %s / %s'
        % (np.round(v_wrist[0].centre, 5) if v_wrist else None,
           np.round(v_fixed[0].centre, 5) if v_fixed else None))

    print()
    print('=' * 74)
    if failures:
        print('  %d check(s) FAILED' % failures)
        print('=' * 74)
        return 1
    print('  Fixed-camera world: correct convention, pose-independent,')
    print('  and identical to the wrist path at the same camera pose.')
    print('  It does NOT verify a real mount: measure FIXED_CAM_POS/TARGET')
    print('  and compare a detected cube against a ruler.')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
