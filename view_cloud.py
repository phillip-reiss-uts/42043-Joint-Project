# view_cloud.py -- offline perception diagnostic. Writes perception.png.
#
# Panels come from the REAL perceive_all call, not a re-implementation: the
# internal helpers are wrapped for the duration of one call and their inputs and
# outputs recorded. So the filter, the clustering and the span-guard verdicts
# shown here are exactly what the task path uses. Nothing is recomputed.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import mujoco

import src.perception as P
from src.model_loader import load_robot_model
from src.camera import RGBDCamera
from src.perception import find_camera, perceive_all
from src.pick_place import ObjectRegistry, reset_scene, place_cube
from src.config import (
    POOL_SIZE, OBJECT_SIZE, OBJECT_HEIGHT, TABLE_HEIGHT, TABLE_CENTRE,
    TABLE_HALF, SELF_FILTER, DEPTH_MAX, SURFACE_BAND,
)

SEED = 0
SIZE = float(np.asarray(OBJECT_SIZE)[0]) * 2.0      # cube edge, 0.060 m
SPAN_HI = SIZE * np.sqrt(2.0) * 1.10                # span guard upper bound
SPAN_LO = SIZE * 0.70                               # span guard lower bound


def build_scene():
    model = load_robot_model()
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    name, cam_id = find_camera(model)
    cam = RGBDCamera(model, name, 640, 480)

    reg = ObjectRegistry(model, POOL_SIZE)
    rng = np.random.default_rng(SEED)
    reset_scene(model, data, reg, rng, viewer=None)
    place_cube(model, data, reg, rng, settle=True)
    mujoco.mj_forward(model, data)
    return model, data, cam, cam_id, reg


def perceive_instrumented(model, data, cam, cam_id):
    # Wrap the real helpers, call the real perceive_all once, restore.
    seen = {}
    saved = (cam.capture, cam.deproject, P._filter_points, P.cluster_objects)

    def spy_capture(d):
        rgb, depth = saved[0](d)
        seen['rgb'], seen['depth'] = rgb, depth
        return rgb, depth

    def spy_deproject(depth, d):
        grid = saved[1](depth, d)
        seen['grid'] = grid
        return grid

    def spy_filter(*a, **k):
        out = saved[2](*a, **k)
        seen['filtered'] = out
        return out

    def spy_cluster(points, *a, **k):
        out = saved[3](points, *a, **k)
        seen['clusters'] = out
        seen['pre_cluster'] = points
        return out

    cam.capture, cam.deproject = spy_capture, spy_deproject
    P._filter_points, P.cluster_objects = spy_filter, spy_cluster
    try:
        dets = perceive_all(model, data, cam, cam_id)
    finally:
        cam.capture, cam.deproject, P._filter_points, P.cluster_objects = saved
    return dets, seen


def main():
    model, data, cam, cam_id, reg = build_scene()
    dets, seen = perceive_instrumented(model, data, cam, cam_id)

    grid = seen.get('grid')
    filt = seen.get('filtered', np.zeros((0, 3)))
    clusters = seen.get('clusters', [])
    raw = grid[np.isfinite(grid).all(axis=-1)] if grid is not None else \
        np.zeros((0, 3))
    cam_pos = np.array(data.cam_xpos[cam_id], float)

    print()
    print('=' * 78)
    print('PERCEPTION DIAGNOSTIC  (one perceive_all call, internals recorded)')
    print('=' * 78)
    print('camera world pos    : %s' % np.round(cam_pos, 4))
    # every object currently on the table, straight from the registry
    truth = {j: reg.pose(data, j) for j in reg.active()}
    print('objects on the table: %d  (%s)'
          % (len(truth), ', '.join('obj %d at %s' % (j, np.round(p[:2], 4))
                                   for j, p in sorted(truth.items()))))
    print()
    print('raw deprojected     : %6d finite points' % len(raw))
    print('after _filter_points: %6d   (self>%.2f m, <%.2f m, z>%.3f m)'
          % (len(filt), SELF_FILTER, DEPTH_MAX, TABLE_HEIGHT + SURFACE_BAND))
    print('clusters            : %6d' % len(clusters))
    print('span guard          : %.4f <= span <= %.4f m'
          % (SPAN_LO, SPAN_HI))
    print()
    print('  %-4s %-8s %-9s %-9s %-8s %-8s %-9s %-20s %s'
          % ('#', 'pts', 'span_x', 'span_y', 'z_min', 'z_max', 'span_ok',
             'verdict', 'cluster centre xy'))
    for i, pts in enumerate(clusters):
        span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        ok = P._span_ok(span)
        big = bool(np.any(span > SPAN_HI))
        small = bool(np.any(span < SPAN_LO))
        if len(pts) < 50:
            verdict = 'REJECT: %d pts < 50' % len(pts)
        elif not ok:
            verdict = 'REJECT: span too %s' % ('big (contamination)' if big
                                               else 'small (clipped view)')
        else:
            verdict = 'ACCEPT'
        c = pts[:, :2].mean(axis=0)
        print('  %-4d %-8d %-9.4f %-9.4f %-8.4f %-8.4f %-9s %-20s %s'
              % (i, len(pts), span[0], span[1],
                 pts[:, 2].min(), pts[:, 2].max(), ok, verdict,
                 np.round(c, 4)))
    print()
    print('  a cube on the table occupies z %.3f to %.3f m'
          % (TABLE_HEIGHT, TABLE_HEIGHT + OBJECT_HEIGHT))
    print()

    # each detection against the NEAREST true object, not a fixed one
    matched = set()
    for d in dets:
        j = min(truth, key=lambda k: np.linalg.norm(truth[k][:2] - d.centre[:2]))
        err = float(np.linalg.norm(truth[j][:2] - d.centre[:2]))
        matched.add(j)
        print('  detection: centre %s  yaw %5.1f deg  span %s  score %.3f'
              % (np.round(d.centre[:2], 4), d.yaw, np.round(d.span, 4), d.score))
        print('             nearest obj %d  xy error %.4f m  z range %.3f to %.3f'
              % (j, err, d.points[:, 2].min(), d.points[:, 2].max()))
    missed = sorted(set(truth) - matched)
    print()
    print('  detected %d of %d objects; missed: %s'
          % (len(matched), len(truth),
             ', '.join('obj %d at %s' % (j, np.round(truth[j][:2], 4))
                       for j in missed) if missed else 'none'))

    # PLOTS
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(seen.get('rgb'))
    ax.set_title('RGB (wrist camera)')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 2)
    im = ax.imshow(seen.get('depth'), cmap='turbo')
    ax.set_title('depth (m)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    for k, (pts, title) in enumerate((
            (raw, '3. raw cloud (%d pts)' % len(raw)),
            (filt, '4. after _filter_points (%d pts)' % len(filt)))):
        a = fig.add_subplot(2, 3, 3 + k)
        if len(pts):
            step = max(1, len(pts) // 40000)     # display only, counts stay exact
            s = pts[::step]
            a.scatter(s[:, 0], s[:, 2], c=s[:, 2], s=0.4, cmap='viridis')
        a.axhline(TABLE_HEIGHT, color='k', ls='--', lw=1.0,
                  label='table top %.3f' % TABLE_HEIGHT)
        a.axhline(TABLE_HEIGHT + SURFACE_BAND, color='m', ls=':', lw=1.0,
                  label='surface band +%.3f' % SURFACE_BAND)
        a.axhline(cam_pos[2], color='r', ls=':', lw=1.0,
                  label='camera z %.3f' % cam_pos[2])
        a.set_xlim(cam_pos[0] - 0.35, cam_pos[0] + 0.35)
        a.set_ylim(TABLE_HEIGHT - 0.05, cam_pos[2] + 0.05)
        a.set_xlabel('world x (m)')
        a.set_ylabel('world z (m)')
        a.set_title(title + '\nside view')
        a.legend(fontsize=6)

    ax = fig.add_subplot(2, 3, 5)
    for i, pts in enumerate(clusters):
        span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        ok = len(pts) >= 50 and P._span_ok(span)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.6,
                   color=('tab:green' if ok else 'tab:red'))
        c = pts[:, :2].mean(axis=0)
        ax.annotate('%d' % i, c, fontsize=8, ha='center')
    for j, p in sorted(truth.items()):
        ax.plot(p[0], p[1], 'k+', ms=8, mew=1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('world x (m)')
    ax.set_ylabel('world y (m)')
    ax.set_title('5. clusters (green = accepted, red = rejected)\n'
                 + str(len(clusters)) + ' found, black + = object truth')

    ax = fig.add_subplot(2, 3, 6)
    ax.add_patch(Rectangle((TABLE_CENTRE[0] - TABLE_HALF[0],
                            TABLE_CENTRE[1] - TABLE_HALF[1]),
                           2 * TABLE_HALF[0], 2 * TABLE_HALF[1],
                           fill=False, ec='0.7', ls='-', lw=1,
                           label='table'))
    for j, p in sorted(truth.items()):
        ax.plot(p[0], p[1], 'k+', ms=14, mew=2,
                label='object truth' if j == min(truth) else None)
        ax.annotate('obj %d' % j, (p[0], p[1]), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    for i, pts in enumerate(clusters):
        span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        if not (len(pts) >= 50 and P._span_ok(span)):
            c = pts[:, :2].mean(axis=0)
            ax.add_patch(Rectangle((c[0] - span[0] / 2, c[1] - span[1] / 2),
                                   span[0], span[1], fill=False,
                                   ec='tab:red', ls=':', lw=1.2))
            ax.annotate('rej %d' % i, c, fontsize=7, color='tab:red')
    for d in dets:
        ax.plot(d.centre[0], d.centre[1], 'o', mfc='none', mec='tab:green',
                ms=14, mew=2, label='fitted centre')
        th = np.radians(d.yaw)
        half = SIZE / 2.0
        ax.plot([d.centre[0] - half * np.cos(th),
                 d.centre[0] + half * np.cos(th)],
                [d.centre[1] - half * np.sin(th),
                 d.centre[1] + half * np.sin(th)], 'tab:green', lw=2)
        ax.add_patch(Rectangle((d.centre[0] - d.span[0] / 2,
                                d.centre[1] - d.span[1] / 2),
                               d.span[0], d.span[1], fill=False,
                               ec='tab:green', ls='--', lw=1.2))
    ax.plot(cam_pos[0], cam_pos[1], 'r*', ms=13, label='camera')
    ax.set_aspect('equal')
    ax.set_xlabel('world x (m)')
    ax.set_ylabel('world y (m)')
    ax.set_title('6. decision: fitted centre, yaw line, measured span\n'
                 'guard %.4f-%.4f m   green=accepted  red=rejected'
                 % (SPAN_LO, SPAN_HI))
    ax.legend(fontsize=6, loc='best')

    fig.suptitle('Perception pipeline -- from one real perceive_all() call')
    fig.tight_layout()
    fig.savefig('perception.png', dpi=110)
    cam.renderer.close()
    print()
    print('wrote perception.png')


if __name__ == '__main__':
    main()
