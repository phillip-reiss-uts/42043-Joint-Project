# Headless test harness.
#
# Runs the closed-loop approach repeatedly against randomised spawns, optionally
# with the object walking, and records the servo path, the object's own path and
# the estimate error at every viewer sync. Writes a summary table plus graphs.
#
#   python run_tests.py                       # 6 trials, static object, KF on
#   python run_tests.py --trials 8 --band medium
#   python run_tests.py --kf both --band fast --trials 4
#
# Everything is headless: the viewer is replaced by a recorder that samples
# state on the sync calls the servo and the arm already make.

import argparse
import csv
import io
import contextlib
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mujoco

from src.model_loader import load_robot_model
from src.camera import RGBDCamera
from src.camera import find_camera
from src.arm import Arm
from src.servo import PBVSServo
from src.pick_place import ObjectRegistry, reset_scene, place_cube
from src.pick import search_for_object
from src.perception import rotation_with_yaw
from src.world_sim import SimObjectWorld
from src.kinematics_mujoco import MujocoKinematics
from src.robot_backend_mujoco import MujocoBackend
from src import target_motion as TM
from src.config import (
    POOL_SIZE, SERVO_STANDOFF, SERVO_MAX_ITERS, SERVO_V_TRANSIT,
    SERVO_V_DESCENT, SERVO_PERCEIVE_EVERY, SERVO_OCCLUDE_RATIO,
    SERVO_OCCLUDE_RECOVER, SERVO_OCCLUDE_FAILS, SERVO_RETREAT_STEP,
    SERVO_RETREAT_MAX, SERVO_RETREAT_V,
    GRIP_CLOSED, R_DOWN, LIFT_Z, STEPS_CLOSE, STEPS_RELEASE,
    GRASP_LIFT_RECHECK, GRASP_CLOSED_Q, GRASP_FORCE_MIN,
    TABLE_HEIGHT, OBJECT_HEIGHT, FEED, ARM_DOF,
)


class Recorder:
    # Stands in for the viewer handle: samples state, draws nothing.

    def __init__(self, model, data, arm, reg, index, servo, world=None):
        self.model = model
        self.data = data
        self.arm = arm
        self.reg = reg
        self.index = index
        self.servo = servo
        self.world = world
        self.overlay = None                  # so the servo's lookups no-op
        self.t, self.ee, self.obj, self.est, self.truth = [], [], [], [], []
        self.raw_est, self.span_ratio = [], []

    def sync(self):
        d = self.data
        # NOTE: the mover is NOT stepped here. sync runs every SYNC_STEPS
        # physics steps, so driving the object from it would move the cube at
        # 1/SYNC_STEPS of its true speed. The world is ticked per physics step
        # by the servo (PBVSServo._step_world) and by Arm's step_hook.
        self.t.append(float(d.time))
        self.ee.append(np.array(d.xpos[self.arm.base_id], float))
        p = self.reg.pose(d, self.index)
        self.obj.append(np.array(p, float))
        self.truth.append(np.array(p, float))
        est = self.servo.p_est
        self.est.append(np.array(est, float) if est is not None else np.full(3, np.nan))
        self.raw_est.append(np.array(self.servo.raw_est, float)
                            if self.servo.raw_est is not None else np.full(3, np.nan))
        self.span_ratio.append(float(self.servo.span_ratio))

    def series(self):
        return {k: np.array(getattr(self, k)) for k in
                ('t', 'ee', 'obj', 'est', 'truth', 'raw_est', 'span_ratio')}


def run_trial(model, camera, cam_id, seed, band, use_kf, args=None):
    # One full approach, recorded.
    TM.set_band(band)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    backend = MujocoBackend(model, data, ARM_DOF)
    arm = Arm(model, data,
              kinematics=MujocoKinematics(model, backend.tool_body_id, ARM_DOF),
              backend=backend)
    reg = ObjectRegistry(model, POOL_SIZE)
    rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        reset_scene(model, data, reg, rng, viewer=None)
        place_cube(model, data, reg, rng, settle=True)
    mujoco.mj_forward(model, data)

    idx = reg.active()[0]
    p0 = np.array(reg.pose(data, idx), float)

    world = SimObjectWorld(model, data, camera, cam_id, reg)
    mover = TM.TargetMover(model, data, reg, idx, rng=rng)
    world.set_mover(mover)
    sv = PBVSServo(model, data, arm, world, smooth=1.0, use_kf=use_kf,
                   perceive_every=SERVO_PERCEIVE_EVERY,
                   v_max=SERVO_V_TRANSIT, v_descent=SERVO_V_DESCENT,
                   occlude_ratio=SERVO_OCCLUDE_RATIO,
                   occlude_recover=SERVO_OCCLUDE_RECOVER,
                   occlude_fails=SERVO_OCCLUDE_FAILS,
                   retreat_step=SERVO_RETREAT_STEP,
                   retreat_max=SERVO_RETREAT_MAX,
                   retreat_v=SERVO_RETREAT_V)
    rec = Recorder(model, data, arm, reg, idx, sv, world=world)

    # Emulate the operator: let the object walk for a while, then stop it. The
    # robot only commits once it has watched the target hold still, so without
    # a stop the trial can never complete -- it just chases to the track cap.
    move_s = 0.0 if args is None else getattr(args, 'move_s', 0.0)
    sabotage_mm = (0.0 if args is None
                   else float(getattr(args, 'sabotage', 0.0)))
    t_start = float(data.time)
    _orig_step = mover.step

    def _step(d):
        # Emulate the operator: shut the band off after move_s seconds.
        if move_s and d.time - t_start >= move_s:
            TM.set_band(0)
        return _orig_step(d)

    mover.step = _step

    def reacquire():
        # same re-acquire the task path uses, so the harness exercises it too
        def perceive():
            return [True] if sv.observe() else []
        return search_for_object(model, data, arm, perceive, viewer=rec,
                                 stop=lambda: False, wide=True)

    sv.observe()
    t0 = time.perf_counter()
    ok_approach, iters, err, hover = sv.approach(
        standoff=SERVO_STANDOFF, viewer=rec, coarse=False,
        max_iters=SERVO_MAX_ITERS, check_occlusion=True, recover=True,
        search=reacquire)
    wall_approach = time.perf_counter() - t0

    # Geometry at the moment the approach finished, before the jaws move.
    mujoco.mj_forward(model, data)
    p_end = np.array(reg.pose(data, idx), float)
    grip = np.array(data.xpos[arm.base_id], float)
    gap_mm = float(np.linalg.norm(grip[:2] - p_end[:2]) * 1e3)

    # Close and verify, exactly as pick_one does. Scoring `ok` off the approach
    # alone reports success whenever the servo converges, which it does just as
    # happily over empty table.
    g = {'state': 'not attempted', 'held': False, 'q': float('nan'),
         'force': float('nan'), 'vel': float('nan')}
    rechecked = False
    lifted_ok = False
    if ok_approach:
        R = rotation_with_yaw(sv.yaw_est, R_DOWN)
        q_lift = arm.solve_ik(grip + np.array([0.0, 0.0, LIFT_Z]),
                              data.qpos[:ARM_DOF], R_target=R)
        if sabotage_mm:
            # Deliberate miss, so the harness exercises the failure path rather
            # than only ever confirming a pass. The cube steps aside at the
            # instant the jaws close; the arm drives to the pose it had already
            # committed to and shuts on empty table. Nothing tells the robot
            # this happened -- the whole point is that only the jaws can tell.
            p_sab = np.array(reg.pose(data, idx), float)
            reg.teleport(data, idx,
                         [p_sab[0] + sabotage_mm * 1e-3, p_sab[1], p_sab[2]],
                         quat=(1.0, 0.0, 0.0, 0.0))
            mujoco.mj_forward(model, data)
        arm.close_gripper(STEPS_CLOSE, rec)
        g = arm.grasp_state()
        if not g['held']:
            # lift clear and read again, as the task does
            rechecked = True
            q_re = arm.solve_ik(grip + np.array([0.0, 0.0, GRASP_LIFT_RECHECK]),
                                data.qpos[:ARM_DOF], R_target=R)
            arm.move(q_re, GRIP_CLOSED, rate=FEED, viewer=rec)
            g = arm.grasp_state()
        if g['held']:
            # Couple the object to the jaws for the carry, exactly as
            # pick.grasp_and_lift does -- without this the harness carries the
            # cube with rigid pads only, which is the failure it is meant to
            # measure around, not reproduce.
            arm.attach_object('obj_%d' % idx)
            arm.move(q_lift, GRIP_CLOSED, rate=FEED, viewer=rec)
            lifted_ok = bool(reg.pose(data, idx)[2]
                             > TABLE_HEIGHT + OBJECT_HEIGHT / 2.0)
        else:
            arm.open_gripper(STEPS_RELEASE, rec)
    wall = time.perf_counter() - t0

    ok = bool(ok_approach and g['held'] and lifted_ok)
    s = rec.series()

    # A trial that aborts before the servo ever syncs records nothing, and an
    # empty series is 1-D, so the norm below raised IndexError and hid the real
    # outcome. Report the empty trial instead of crashing the whole harness.
    if s['t'].size == 0:
        print('  %-5d %-6s %-4s  ABORTED before any observation '
              '(approach_ok=%s)' % (seed, TM.band_names()[band],
                                    'on' if use_kf else 'off', ok_approach))
        nan = float('nan')
        return dict(
            seed=seed, band=TM.band_names()[band], kf=bool(use_kf),
            ok=False, sabotage_mm=float(sabotage_mm),
            approach_ok=bool(ok_approach), grasp_state=g['state'],
            grasp_held=bool(g['held']), rechecked=bool(rechecked),
            lifted=bool(lifted_ok),
            grasp_q=float(g['q']), grasp_force=float(g['force']),
            grasp_vel=float(g['vel']),
            iters=0, wall_s=wall, wall_approach_s=nan,
            grasp_gap_mm=nan, obj_shift_mm=nan, obj_path_mm=0.0,
            obj_turns=0, occluded=False, recovered=False,
            kf_updates=0, kf_rejects=0,
            est_err_mean_mm=nan, est_err_max_mm=nan,
            raw_err_mean_mm=nan, raw_err_max_mm=nan,
            dist_start_m=nan, dist_min_m=nan,
            series=s,
        )

    est_err = np.linalg.norm(s['est'][:, :2] - s['truth'][:, :2], axis=1)
    raw_err = np.linalg.norm(s['raw_est'][:, :2] - s['truth'][:, :2], axis=1)
    dist = np.linalg.norm(s['ee'] - s['obj'], axis=1)

    return dict(
        seed=seed, band=TM.band_names()[band], kf=bool(use_kf),
        ok=ok, sabotage_mm=float(sabotage_mm),
        approach_ok=bool(ok_approach), grasp_state=g['state'],
        grasp_held=bool(g['held']), rechecked=bool(rechecked),
        lifted=bool(lifted_ok),
        grasp_q=float(g['q']), grasp_force=float(g['force']),
        grasp_vel=float(g['vel']),
        iters=int(iters), wall_s=wall, wall_approach_s=wall_approach,
        grasp_gap_mm=gap_mm,
        obj_shift_mm=float(np.linalg.norm(p_end[:2] - p0[:2]) * 1e3),
        obj_path_mm=float(mover.moved * 1e3),
        obj_turns=int(mover.turned),
        occluded=bool(sv.occluded), recovered=bool(sv.recovered),
        kf_updates=int(sv.kf.updates) if sv.kf is not None else 0,
        kf_rejects=int(sv.kf.rejects) if sv.kf is not None else 0,
        est_err_mean_mm=float(np.nanmean(est_err) * 1e3),
        est_err_max_mm=float(np.nanmax(est_err) * 1e3),
        raw_err_mean_mm=float(np.nanmean(raw_err) * 1e3),
        raw_err_max_mm=float(np.nanmax(raw_err) * 1e3),
        dist_start_m=float(dist[0]) if len(dist) else float('nan'),
        dist_min_m=float(np.nanmin(dist)) if len(dist) else float('nan'),
        series=s,
    )


# PLOTS
def plot_paths(runs, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in runs:
        s = r['series']
        if not len(s['t']):
            # Trial aborted before any observation; nothing to draw.
            continue
        lbl = '%s kf=%s seed %d' % (r['band'], 'on' if r['kf'] else 'off', r['seed'])
        ax.plot(s['ee'][:, 0], s['ee'][:, 1], '-', lw=1.4, label=lbl)
        ax.plot(s['obj'][0, 0], s['obj'][0, 1], 'o', ms=6, mfc='none',
                mec='k', mew=1.2)
        ax.plot(s['obj'][-1, 0], s['obj'][-1, 1], 'x', ms=8, color='k')
        if r['obj_path_mm'] > 1.0:
            ax.plot(s['obj'][:, 0], s['obj'][:, 1], ':', lw=1.0, color='0.5')
    ax.plot([], [], 'o', mfc='none', mec='k', label='object start')
    ax.plot([], [], 'x', color='k', label='object end')
    ax.set_aspect('equal')
    ax.set_xlabel('world x (m)')
    ax.set_ylabel('world y (m)')
    ax.set_title('Gripper path per trial (dotted = object path)')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'servo_paths.png'), dpi=120)
    plt.close(fig)


def plot_distance(runs, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in runs:
        s = r['series']
        if not len(s['t']):
            continue
        d = np.linalg.norm(s['ee'] - s['obj'], axis=1)
        ax.plot(s['t'] - s['t'][0], d, lw=1.3,
                label='%s kf=%s seed %d' % (r['band'], 'on' if r['kf'] else 'off',
                                            r['seed']))
    ax.set_xlabel('sim time since approach start (s)')
    ax.set_ylabel('gripper to object distance (m)')
    ax.set_title('Approach profile: distance closure')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'distance_closure.png'), dpi=120)
    plt.close(fig)


def plot_error(runs, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in runs:
        s = r['series']
        if not len(s['t']):
            continue
        e = np.linalg.norm(s['est'][:, :2] - s['truth'][:, :2], axis=1) * 1e3
        ax.plot(s['t'] - s['t'][0], e, lw=1.3,
                label='%s kf=%s seed %d' % (r['band'], 'on' if r['kf'] else 'off',
                                            r['seed']))
    ax.set_xlabel('sim time since approach start (s)')
    ax.set_ylabel('|estimate - truth| xy (mm)')
    ax.set_title('Servo estimate error over the approach')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'estimate_error.png'), dpi=120)
    plt.close(fig)


def plot_summary(runs, outdir):
    fig, axes = plt.subplots(1, 5, figsize=(21, 4))
    labels = ['%s/%s\ns%d' % (r['band'], 'kf' if r['kf'] else 'raw', r['seed'])
              for r in runs]
    x = np.arange(len(runs))

    axes[0].bar(x, [r['grasp_gap_mm'] for r in runs], color='tab:blue')
    axes[0].set_ylabel('grasp gap (mm)')
    axes[0].set_title('Final gripper-to-object offset')

    axes[1].bar(x, [r['obj_path_mm'] for r in runs], color='tab:orange')
    axes[1].set_ylabel('object path (mm)')
    axes[1].set_title('How far the object travelled')

    axes[2].bar(x - 0.2, [r['est_err_mean_mm'] for r in runs], width=0.4,
                color='tab:green', label='filtered')
    axes[2].bar(x + 0.2, [r['raw_err_mean_mm'] for r in runs], width=0.4,
                color='0.6', label='raw')
    axes[2].set_ylabel('mean estimate error (mm)')
    axes[2].set_title('Vision error: filtered vs raw')
    axes[2].legend(fontsize=8)

    # grasp verdict from the gripper's own registers: anything at or above the
    # dashed line closed on empty jaws
    axes[3].bar(x, [r['grasp_q'] for r in runs], color='tab:purple')
    axes[3].axhline(GRASP_CLOSED_Q, color='r', ls='--', lw=1.2,
                    label='closed on nothing')
    axes[3].set_ylabel('finger position at close')
    axes[3].set_title('Grasp check: jaw opening')
    axes[3].legend(fontsize=8)

    axes[4].bar(x, [r['grasp_force'] for r in runs], color='tab:red')
    axes[4].axhline(GRASP_FORCE_MIN, color='k', ls='--', lw=1.2,
                    label='grip threshold')
    axes[4].set_ylabel('actuator force')
    axes[4].set_title('Grasp check: grip force')
    axes[4].legend(fontsize=8)

    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=6)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--band', default='off',
                    help="off|slow|medium|fast, or an index")
    ap.add_argument('--kf', default='on', help='on|off|both')
    ap.add_argument('--move-s', type=float, default=8.0,
                    help='seconds to let the object walk before stopping it;'
                         ' 0 leaves it moving forever (trial cannot complete)')
    ap.add_argument('--sabotage', type=float, default=0.0,
                    help='mm to displace the cube sideways at the instant the jaws\n'
                         'close, forcing a deliberate miss so the failure path\n'
                         'is exercised. e.g. --sabotage 60')
    ap.add_argument('--outdir', default='test_output')
    args = ap.parse_args()

    names = TM.band_names()
    band = int(args.band) if args.band.isdigit() else names.index(args.band)
    kfs = [True, False] if args.kf == 'both' else [args.kf != 'off']

    os.makedirs(args.outdir, exist_ok=True)
    model = load_robot_model()
    name, cam_id = find_camera(model)
    camera = RGBDCamera(model, name, 320, 240)

    runs = []
    print()
    print('=' * 96)
    print('  HEADLESS SERVO HARNESS   band=%s  kf=%s  trials=%d%s'
          % (names[band], args.kf, args.trials,
             '   SABOTAGE %.0f mm at close' % args.sabotage
             if args.sabotage else ''))
    print('=' * 96)
    print('  %-5s %-7s %-4s %-5s %-8s %-9s %-9s %-8s %-8s %-8s %-11s %s'
          % ('seed', 'band', 'kf', 'grasp', 'approach', 'gap mm', 'jaw q',
             'force', 'obj mm', 'est err', 'occ/rec', 'kf upd/rej'))
    for kf in kfs:
        for i in range(args.trials):
            r = run_trial(model, camera, cam_id, args.seed0 + i, band, kf,
                          args=args)
            runs.append(r)
            print('  %-5d %-7s %-4s %-5s %-8s %-9.2f %-9.3f %-8.2f %-8.1f '
                  '%-8.2f %-11s %d/%d'
                  % (r['seed'], r['band'], 'on' if r['kf'] else 'off',
                     'ok' if r['ok'] else r['grasp_state'][:5],
                     'ok' if r['approach_ok'] else 'FAILED',
                     r['grasp_gap_mm'], r['grasp_q'], r['grasp_force'],
                     r['obj_path_mm'], r['est_err_mean_mm'],
                     '%s/%s' % (r['occluded'], r['recovered']),
                     r['kf_updates'], r['kf_rejects']))
        print()

    gaps = [r['grasp_gap_mm'] for r in runs]
    print('  grasp gap : mean %.2f  max %.2f mm   (n=%d)'
          % (np.mean(gaps), np.max(gaps), len(gaps)))
    print('  approach  : %d/%d complete'
          % (sum(1 for r in runs if r['approach_ok']), len(runs)))
    print('  grasp held: %d/%d verified by the jaws'
          % (sum(1 for r in runs if r['grasp_held']), len(runs)))
    print('  success   : %d/%d picked and lifted (approach AND grasp AND lift)'
          % (sum(1 for r in runs if r['ok']), len(runs)))
    print('  rechecks  : %d trial(s) needed a lift-and-recheck'
          % sum(1 for r in runs if r['rechecked']))

    # summary csv
    fields = [k for k in runs[0] if k != 'series']
    with open(os.path.join(args.outdir, 'summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in fields})

    plot_paths(runs, args.outdir)
    plot_distance(runs, args.outdir)
    plot_error(runs, args.outdir)
    plot_summary(runs, args.outdir)
    camera.renderer.close()
    print('  wrote summary.csv, servo_paths.png, distance_closure.png,')
    print('        estimate_error.png, summary.png  ->  %s/' % args.outdir)


if __name__ == '__main__':
    main()
