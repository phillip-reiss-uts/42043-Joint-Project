"""The opt-in snapshot feature: projection, drawing, saving, and being OFF.

The projection here is the inverse of camera.world_points_from_depth, which is
the single place the depth -> world maths lives. It is tested by ROUND TRIP
through that function rather than against hand-computed pixels: a hand-computed
test would agree with a consistently wrong sign in both places, which is exactly
the bug this is meant to catch (see the K[0,0] sign note in src/liveview.py).

Offline: no camera, no robot, no display. cv2 writes to a temp directory.
"""

import os
import sys


import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src import config as cfg                          # noqa: E402
from src.camera import world_points_from_depth         # noqa: E402
from src.liveview import annotate, depth_image, project, Snapshotter  # noqa: E402
from src.world_real import DepthWorld, FakeDepthSource  # noqa: E402

PASSED = 0
FAILED = 0


def check(name, ok, detail=''):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print('  [ok]   %s' % name)
    else:
        FAILED += 1
        print('  [FAIL] %s' % name)
        if detail:
            print('         %s' % detail)


def _looking_down(pos, target, up_hint):
    # Camera pose in the base frame from a position and a look-at point, in the
    # deprojection's convention: columns [right, up, backward], looking along
    # its own -z.
    pos = np.asarray(pos, float)
    back = -(np.asarray(target, float) - pos)
    back = back / max(float(np.linalg.norm(back)), 1e-12)
    up = np.asarray(up_hint, float)
    if abs(float(np.dot(up, back))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(up, back))) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
    right = np.cross(up, back)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(back, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, up, back
    T[:3, 3] = pos
    return T


def _world():
    # Fake source plus a camera looking straight down from the height that source
    # assumes. cam_world_z is the property that makes the synthetic depth land on
    # the tabletop; a literal here is how the fixture and the decoder disagree.
    src = FakeDepthSource()
    pos = np.array([cfg.TABLE_CENTRE[0], cfg.TABLE_CENTRE[1], src.cam_world_z])
    T = _looking_down(pos, np.array([*cfg.TABLE_CENTRE, cfg.TABLE_HEIGHT]),
                      (0.0, -1.0, 0.0))

    class _FixedWorld(DepthWorld):
        def camera_pose(inner):
            return T[:3, 3].copy(), T[:3, :3].copy()

    return src, _FixedWorld(src, lambda: (np.zeros(3), np.eye(3)))


def test_projection_round_trips_through_the_shipped_maths():
    src, world = _world()
    z16, K = src.capture()
    p_cam_w, R_wc = world.camera_pose()

    # Take the world points the shipped deprojection produces, then project them
    # back and require the pixel they came from.
    pts = world_points_from_depth(z16.astype(float) * 1e-3, K, R_wc, p_cam_w,
                                  cfg.SELF_FILTER, cfg.DEPTH_MAX,
                                  cfg.SURFACE_Z_MIN,
                                  xy_bounds=cfg.TABLE_XY_BOUNDS)
    check('the fixture yields points to round-trip', len(pts) > 100,
          '%d points' % len(pts))

    # Reconstruct which pixel each surviving point came from by projecting it and
    # confirming the depth at that pixel matches the point's depth.
    worst = 0
    for p in pts[::max(1, len(pts) // 40)]:
        uv = project(p, p_cam_w, R_wc, K)
        if uv is None:
            worst = 1e9
            break
        u, v = int(round(uv[0])), int(round(uv[1]))
        if not (0 <= u < z16.shape[1] and 0 <= v < z16.shape[0]):
            worst = 1e9
            break
        # distance along the ray at that pixel must equal the point's distance
        ray = R_wc.T @ (p - p_cam_w)
        worst = max(worst, abs(float(z16[v, u]) / 1000.0 + float(ray[2])))
    check('every sampled point projects onto a pixel carrying its own depth',
          worst < 2e-3, 'worst |depth mismatch| %.4f m' % worst)


def test_projection_rejects_points_behind_the_camera():
    src, world = _world()
    _, K = src.capture()
    p_cam_w, R_wc = world.camera_pose()
    behind = np.array(p_cam_w, float) + 2.0 * np.array(R_wc[:, 2], float)
    check('a point behind the lens returns None (not a wrapped pixel)',
          project(behind, p_cam_w, R_wc, K) is None)


def test_depth_image_flags_dropouts():
    src, _ = _world()
    z16, _K = src.capture()
    img = depth_image(z16)
    check('depth image has the frame shape', img.shape[:2] == z16.shape,
          '%s vs %s' % (img.shape[:2], z16.shape))
    blank = depth_image(np.zeros_like(z16))
    check('zero depth is drawn black', int(blank.max()) == 0)
    check('real depth is not black', int(img.max()) > 0)


def test_annotate_draws_the_detection():
    src, world = _world()
    frame = None
    world.cache_frames = True
    views = world.perceive()
    frame = world.last_perception()
    check('the cache holds the frame the views came from', frame is not None)
    check('a detection was found to draw', len(views) == 1,
          '%d views' % len(views))

    bare = depth_image(frame['z16'])
    drawn = annotate(frame['z16'], frame['K'], frame['p_cam_w'], frame['R_wc'],
                     frame['views'], note='test')
    check('annotated image keeps the frame shape', drawn.shape == bare.shape)
    check('annotating changes pixels (something was drawn)',
          int(np.abs(drawn.astype(int) - bare.astype(int)).sum()) > 0)


def test_snapshotter_writes_readable_files():
    import cv2
    src, world = _world()
    world.cache_frames = True
    world.perceive()
    frame = world.last_perception()
    img = annotate(frame['z16'], frame['K'], frame['p_cam_w'], frame['R_wc'],
                   frame['views'])

    # NOT tempfile: the Windows sandbox denies cleanup of temp directories, so a
    # TemporaryDirectory raises PermissionError on exit and takes the whole test
    # run with it. test_output/ is inside the workspace and is already the home
    # for artefacts.
    tmp = os.path.join(os.path.dirname(HERE), 'test_output', 'liveview_test')
    shots = Snapshotter(os.path.join(tmp, 'nested'))       # must create the dir
    try:
        p1 = shots.save(img)
        p2 = shots.save(img)
        check('files are numbered and distinct', p1 != p2 and shots.n == 2)
        back = cv2.imread(p1, cv2.IMREAD_COLOR)
        check('the written file reads back', back is not None
              and back.shape == img.shape,
              'read %r' % ((None if back is None else back.shape),))
        delta = (0 if back is None else
                 int(np.abs(back.astype(int) - img.astype(int)).max()))
        check('bytes survive the round trip (PNG is lossless)', delta == 0,
              'max pixel delta %d' % delta)
    finally:
        for p in (os.path.join(tmp, 'nested', 'snap_001.png'),
                  os.path.join(tmp, 'nested', 'snap_002.png')):
            try:
                os.remove(p)
            except OSError:
                pass


def test_the_feature_is_off_by_default():
    # THE BYPASS REQUIREMENT: with the flag absent nothing is cached, and a world
    # that never sets cache_frames must behave exactly as it did before the
    # feature existed -- same detections, no retained frame.
    _, world = _world()
    views = world.perceive()
    check('last_perception() is None when the cache was never enabled',
          world.last_perception() is None)
    check('perception itself is unaffected with the cache off', len(views) == 1,
          '%d views' % len(views))

    # And when it IS armed, switching it off again stops NEW frames landing: the
    # old dict stays put rather than being cleared, which matters because the
    # operator may want the last frame after the fact.
    world.cache_frames = True
    world.perceive()
    first = world.last_perception()
    check('switching it on caches', first is not None)
    world.cache_frames = False
    world.perceive()
    check('switching it off stops caching (the cached frame is not replaced)',
          world.last_perception() is first)


def test_one_forward_kinematics_call_per_look():
    # The cache must not cost a second FK lookup: deproject() takes the pose the
    # caller already had.
    _, world = _world()
    calls = []
    real = world.camera_pose
    world.camera_pose = lambda: (calls.append(1), real())[1]

    world.cache_frames = False
    world.perceive()
    off = len(calls)

    calls.clear()
    world.cache_frames = True
    world.perceive()
    on = len(calls)

    check('cache off costs one FK call', off == 1, '%d calls' % off)
    check('cache on still costs one FK call', on == 1, '%d calls' % on)


def main():
    print('=' * 70)
    print('SNAPSHOT FEATURE (src/liveview.py) -- offline')
    print('=' * 70)
    for fn in (test_projection_round_trips_through_the_shipped_maths,
               test_projection_rejects_points_behind_the_camera,
               test_depth_image_flags_dropouts,
               test_annotate_draws_the_detection,
               test_snapshotter_writes_readable_files,
               test_the_feature_is_off_by_default,
               test_one_forward_kinematics_call_per_look):
        print('\n%s' % fn.__name__.replace('test_', '').replace('_', ' '))
        fn()
    print('\n%d passed, %d failed' % (PASSED, FAILED))
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
