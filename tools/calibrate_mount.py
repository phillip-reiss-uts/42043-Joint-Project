#!/usr/bin/env python3
"""Wrist-mount and grasp calibration, as a menu.

Every number here is stated from the TOOL FLANGE, in metres, except the jaw
widths, which rg_grip() and the datasheet both speak in millimetres.

Edits are held in memory and written only when you say so:
    s   write src/config.py
    w   write calib_<stamp>.json (nothing in the tree is touched)
    r   discard the session

Steps, in the order they should be run:

  1 table height    tabletop above the base
  2 camera height   250 mm tool z, ruler from the tabletop to the lens
  3 translation     x, then y, by jogging the tool onto the cube
  4 angle           the cube's position in the TOOL frame from several poses
  5 pick height     descend to a chosen tool height, accept or retry
  6 grip width      close, lift 20 mm, put back; accept or retry

A snapshot of every perception is written next to the record, because the
[PERC] numbers say a look failed and the picture says why.

    python tools/calibrate_mount.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg                  # noqa: E402
from src.run_log import start_run_log          # noqa: E402
from src.arm_dynamics import rot_from_quat     # noqa: E402

CONFIG_PATH = os.path.join(ROOT, 'src', 'config.py')

# Step 2's tool height above the tabletop, m. Low enough to measure with a ruler
# across the gripper, high enough to clear it.
CAM_HEIGHT_TOOL_Z = 0.250

# Step 6: how far the jaws lift to prove they hold the cube, m.
GRIP_TEST_LIFT = 0.020

# Step 3 / 5: what counts as "centred" and how many tries before giving up.
OFFSET_TOL = 0.005
MAX_TRIES = 6


# ---------------------------------------------------------------- session
class Session:
    # The values being calibrated. Starts from config; only save() touches the
    # file, so the session and the file can legitimately disagree.
    def __init__(self):
        self.table_height = float(cfg.TABLE_HEIGHT)
        self.cam_pos = list(map(float, cfg.CAM_MOUNT_POS))
        self.cam_quat = list(map(float, cfg.CAM_MOUNT_QUAT))
        self.pick_z_above_table = float(cfg.PICK_Z_ABOVE_TABLE)
        self.close_width_mm = float(cfg.RG2_CLOSE_WIDTH_MM)
        self.notes = {}          # step -> {residual / paths / whatever}
        self.stamp = time.strftime('%Y-%m-%d_%H%M')

    def mount(self):
        # FLANGE -> camera, from the session values.
        T = np.eye(4)
        T[:3, :3] = rot_from_quat(tuple(self.cam_quat))
        T[:3, 3] = np.array(self.cam_pos, float)
        return T

    def pick_z(self):
        return self.table_height + self.pick_z_above_table

    def changed(self):
        pairs = [
            ('TABLE_HEIGHT', cfg.TABLE_HEIGHT, self.table_height),
            ('CAM_POS', tuple(cfg.CAM_MOUNT_POS), tuple(self.cam_pos)),
            ('CAM_QUAT', tuple(cfg.CAM_MOUNT_QUAT), tuple(self.cam_quat)),
            ('PICK_Z_ABOVE_TABLE', cfg.PICK_Z_ABOVE_TABLE,
             self.pick_z_above_table),
            ('RG2_CLOSE_WIDTH_MM', cfg.RG2_CLOSE_WIDTH_MM,
             self.close_width_mm),
        ]
        return [(n, old, new) for n, old, new in pairs
                if not _same(old, new)]

    def header(self):
        print()
        print('=' * 74)
        print('WRIST MOUNT CALIBRATION            %s   (all metres unless _MM)'
              % self.stamp)
        print('=' * 74)
        rows = [
            ('TABLE_HEIGHT', self.table_height, cfg.TABLE_HEIGHT),
            ('CAM_POS', tuple(round(v, 5) for v in self.cam_pos),
             tuple(round(v, 5) for v in tuple(cfg.CAM_MOUNT_POS))),
            ('CAM_QUAT', tuple(round(v, 5) for v in self.cam_quat),
             tuple(round(v, 5) for v in tuple(cfg.CAM_MOUNT_QUAT))),
            ('PICK_Z_ABOVE_TABLE', self.pick_z_above_table,
             cfg.PICK_Z_ABOVE_TABLE),
            ('RG2_CLOSE_WIDTH_MM', self.close_width_mm, cfg.RG2_CLOSE_WIDTH_MM),
        ]
        for name, now, filed in rows:
            flag = '' if _same(now, filed) else '   CHANGED'
            print('  %-20s %-42s%s' % (name, _fmt(now), flag))
        print('  PICK_Z               %-42s (table + above)'
              % _fmt(round(self.pick_z(), 5)))
        n = len(self.changed())
        print('  unsaved changes      %d' % n)
        print('=' * 74)


def _same(a, b, tol=1e-9):
    try:
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                                atol=tol, rtol=0))
    except (TypeError, ValueError):
        return a == b


def _fmt(v):
    if isinstance(v, (tuple, list)):
        return '(' + ', '.join('%.5f' % x for x in v) + ')'
    return '%.5f' % v


# ---------------------------------------------------------------- hardware
class Cell:
    # One RTDE session and one depth source, opened lazily: steps 1 and 2 need
    # neither.
    def __init__(self, ip):
        self.ip = ip
        self._backend = None
        self._depth = None
        self._kin = None

    def kin(self):
        if self._kin is None:
            from src.arm_dynamics import ArmDynamics
            from src.kinematics_ur import URKinematics
            dyn = ArmDynamics.from_file(
                os.path.join(ROOT, 'models', 'ur3e', 'arm_dynamics.json'))
            # FLANGE frame: every number this tool writes is flange-relative.
            self._kin = URKinematics(
                dyn, (np.full(cfg.ARM_DOF, cfg.UR_SERVO_KP),
                      np.full(cfg.ARM_DOF, cfg.UR_SERVO_KV)),
                tool_pos=(0.0, 0.0, 0.0),
                tool_quat=(1.0, 0.0, 0.0, 0.0),
                root_yaw=cfg.UR_ROOT_YAW)
        return self._kin

    def backend(self):
        if self._backend is None:
            from src.robot_backend_ur import connect
            print('[CAL] connecting to %s ...' % self.ip)
            self._backend = connect(self.ip, activate_gripper=True,
                                    dt=cfg.UR_CYCLE_DT,
                                    lookahead_time=cfg.UR_LOOKAHEAD_TIME,
                                    gain=cfg.UR_SERVO_GAIN,
                                    force_n=cfg.RG2_GRIP_FORCE_N)
        return self._backend

    def arm(self):
        from src.arm import Arm
        return Arm(None, None, kinematics=self.kin(), backend=self.backend())

    def depth(self):
        if self._depth is None:
            from src.world_real import RealSenseSource
            self._depth = RealSenseSource()
        return self._depth

    def world(self, session, cache=False):
        # Built per call so a step can change the mount between looks.
        return _MountedWorld(self.depth(), self.kin, self.backend,
                             session.mount(), cache)

    def tool_pose(self):
        p, _R = self.kin().pose(self.backend().joint_positions())
        return np.array(p, float)

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        if self._depth is not None:
            self._depth.close()
            self._depth = None


class _MountedWorld:
    # RealObjectWorld with a mount that can change between steps, and with the
    # frame cache always on so a snapshot and a tool-frame point are available.
    def __init__(self, depth, kin, backend, mount, cache):
        from src.world_real import RealObjectWorld
        self._w = RealObjectWorld(
            depth, flange_pose=lambda: kin().pose(backend().joint_positions()),
            T_flange_cam=mount)
        self._w.cache_frames = cache

    def perceive(self):
        return self._w.perceive()

    def camera_pose(self):
        return self._w.camera_pose()

    def gripper_pose(self):
        return self._w.gripper_pose()

    def last_perception(self):
        return self._w.last_perception()

    def close(self):
        pass


# ---------------------------------------------------------------- helpers
def _ask_float(prompt, default):
    while True:
        raw = input('%s [%.4f]: ' % (prompt, default)).strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print('  not a number')


def _ask_yes(prompt, default=True):
    raw = input('%s [%s]: ' % (prompt, 'Y/n' if default else 'y/N')).strip()
    if not raw:
        return default
    return raw.strip().lower().startswith('y')


def _grip_pos(width_mm):
    # Jaw width in mm -> the 0..255 command space the arm and the backend use.
    span = cfg.GRIP_CLOSED - cfg.GRIP_OPEN
    lo, hi = cfg.RG2_OPEN_WIDTH_MM, cfg.RG2_CLOSE_WIDTH_MM
    frac = (width_mm - lo) / (hi - lo) if hi != lo else 0.0
    return float(np.clip(cfg.GRIP_OPEN + span * frac,
                         cfg.GRIP_OPEN, cfg.GRIP_CLOSED))


def _seed(q):
    # Every arm joint is revolute, so q and q + 2*pi*k are the same pose. The
    # IK clips to +-2*pi ABSOLUTE, so a joint sitting on that boundary (seen on
    # the cell: q6 = -6.283) cannot be moved by the solver at all. Wrap the seed
    # into (-pi, pi] first; the pose is unchanged.
    q = np.asarray(q, float)[:cfg.ARM_DOF]
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def _move_tool(cell, session, xy, z, grip=cfg.GRIP_OPEN, rate=None):
    # One IK solve to a flange pose, then a plain joint move. Keeps the current
    # orientation, matching what the servo holds.
    arm = cell.arm()
    q_now = np.asarray(cell.backend().joint_positions(), float)
    _p, R = cell.kin().pose(q_now)
    q = arm.solve_ik([xy[0], xy[1], z], _seed(q_now),
                     R_target=R, max_joint=2.0 * np.pi)
    p_ik, _ = cell.kin().pose(q[:cfg.ARM_DOF])
    miss = np.linalg.norm(p_ik - np.array([xy[0], xy[1], z]))
    if miss > 2e-3:
        # The damped solver can pin itself in a straight-elbow/straight-wrist
        # basin when seeded from a joint vector already near one. The pose is
        # reachable from the home seed, so retry there once and log it.
        q_home = arm.solve_ik([xy[0], xy[1], z],
                              np.asarray(cfg.UR_HOME_Q, float),
                              R_target=R, max_joint=2.0 * np.pi)
        p_home, _ = cell.kin().pose(q_home[:cfg.ARM_DOF])
        if np.linalg.norm(p_home - np.array([xy[0], xy[1], z])) <= 2e-3:
            print('  IK stalled from the current joints (%.1f mm short); '
                  're-solved from home' % (miss * 1e3))
            q = q_home
            miss = 0.0
    if miss > 2e-3:
        print('  IK could not reach that pose -- not moving')
        print('  from flange (%+.4f, %+.4f, %+.4f)' % tuple(_p))
        print('  to   flange (%+.4f, %+.4f, %+.4f)' % (xy[0], xy[1], z))
        print('  solved to   (%+.4f, %+.4f, %+.4f), miss %.1f mm'
              % (p_ik[0], p_ik[1], p_ik[2],
                 np.linalg.norm(p_ik - np.array([xy[0], xy[1], z])) * 1e3))
        print('  joints now  %s' % np.round(q_now[:cfg.ARM_DOF], 3))
        return False
    arm.move(q, grip, rate=rate or cfg.FEED)
    return True


def _move_pose(cell, p, R, grip=cfg.GRIP_OPEN, rate=None, settle=0.30):
    # Move to an explicit flange pose, holding the supplied orientation, and
    # hold it for a moment so the next look is not taken mid-motion.
    arm = cell.arm()
    p = np.asarray(p, float)
    q = arm.solve_ik(p, _seed(cell.backend().joint_positions()),
                     R_target=R, max_joint=2.0 * np.pi)
    p_ik, _ = cell.kin().pose(q[:cfg.ARM_DOF])
    if np.linalg.norm(p_ik - p) > 2e-3:
        print('  IK could not reach that pose -- not moving')
        return False
    arm.move(q, grip, rate=rate or cfg.FEED)
    for _ in range(max(1, int(settle / cell.backend().dt))):
        cell.backend().command(q, grip)
        cell.backend().step()
    return True


def _look(cell, session, note, snaps, tag):
    # One perception with the session mount. Returns the TargetView list, and
    # writes an annotated snapshot.
    w = cell.world(session, cache=True)
    views = w.perceive()
    fr = w.last_perception()
    if fr is not None and snaps is not None:
        try:
            from src.liveview import annotate
            img = annotate(fr['z16'], fr['K'], fr['p_cam_w'], fr['R_wc'],
                           fr['views'], note=note,
                           tool_pose=cell.tool_pose())
            path = snaps.save(img, prefix=tag)
            print('  snapshot: %s' % path)
        except Exception as exc:          # a snapshot must never stop the step
            print('  (snapshot failed: %s)' % exc)
    return [v for v in views if v.ok]


def _best(views, near=None):
    if not views:
        return None
    if near is None:
        return views[0]
    return min(views, key=lambda v: float(np.linalg.norm(
        np.asarray(v.centre)[:2] - np.asarray(near)[:2])))


# ------------------------------------------------------------------ steps
def step_table(cell, session, snaps, log):
    print()
    print('1  TABLE HEIGHT -- tabletop above the robot base plane')
    v = _ask_float('  table height, m', session.table_height)
    session.table_height = v
    log['table_height'] = v


def step_camera(cell, session, snaps, log):
    print()
    print('2  CAMERA HEIGHT -- the tool parks at %.0f mm and you measure the'
          % (CAM_HEIGHT_TOOL_Z * 1e3))
    print('   ruler from the TABLETOP to the centre of the lens.')
    p = cell.tool_pose()
    target_z = session.table_height + CAM_HEIGHT_TOOL_Z
    print('  tool now at z %.3f, moving to %.3f (table + %.0f mm)'
          % (p[2], target_z, CAM_HEIGHT_TOOL_Z * 1e3))
    if not _move_tool(cell, session, p[:2], target_z):
        return
    p = cell.tool_pose()
    print('  tool parked at z %.4f -- gripper top is clear for the ruler' % p[2])
    tool_above = p[2] - session.table_height
    lens = _ask_float('  tabletop to lens, m',
                      CAM_HEIGHT_TOOL_Z - session.cam_pos[2])
    # CAM_POS is flange -> camera with +z down the approach, so a lens BELOW
    # tool0 is a POSITIVE offset: tool height minus lens height.
    session.cam_pos[2] = tool_above - lens
    print('  CAM_POS_z = %.5f (tool %.3f - lens %.3f)'
          % (session.cam_pos[2], tool_above, lens))
    log['camera_height'] = dict(lens_m=lens, tool_z=p[2],
                                cam_pos_z=session.cam_pos[2])


def step_home(cell, session, snaps, log):
    print()
    print('H  HOME -- send the arm to the home configuration')
    if not _ask_yes('  move to home?', True):
        return
    arm = cell.arm()
    print('  moving to home ...')
    arm.move(cfg.UR_HOME_Q, cfg.GRIP_OPEN, rate=cfg.RAPID)
    q = cell.backend().joint_positions()
    p, _R = cell.kin().pose(q)
    print('  joints now  %s' % np.round(q[:cfg.ARM_DOF], 3))
    print('  flange now  (%+.4f, %+.4f, %+.4f)' % tuple(p))
    log['home'] = dict(joints=[float(v) for v in q[:cfg.ARM_DOF]],
                       flange=[float(v) for v in p])


def step_translation(cell, session, snaps, log):
    print()
    print('3  TRANSLATION -- x then y. The tool moves ONLY the axis under test,')
    print('   by exactly the number of millimetres you type. Type a signed')
    print('   distance, watch the jaws, type the next one; press Enter when the')
    print('   jaws straddle the cube. Nothing is inferred between entries, so')
    print('   the same sign always moves the same way.')
    z = session.pick_z() + 0.150          # look from above the pick pose
    p = cell.tool_pose()
    if not _move_tool(cell, session, p[:2], z):
        return
    out = {}
    for axis, name in ((0, 'x'), (1, 'y')):
        print()
        print('   -- %s --' % name)
        start = session.cam_pos[axis]
        input('   place the cube in view, then Enter > ')
        views = _look(cell, session, 'step 3 %s' % name, snaps,
                      'step3_%s' % name)
        v = _best(views, near=cell.tool_pose()[:2])
        if v is None:
            print('   no cube seen -- skipping %s' % name)
            out[name] = dict(residual_m=None, aborted=True)
            continue
        c = np.asarray(v.centre, float)
        # First move: onto the perceived centre, one axis only. Everything
        # after this is a plain relative jog, with no perception in the loop.
        p = cell.tool_pose()
        target = [p[0], p[1]]
        target[axis] = c[axis]
        print('   cube reads (%+.4f, %+.4f); moving %s to %+.4f'
              % (c[0], c[1], name, target[axis]))
        if not _move_tool(cell, session, target, z):
            return

        total = 0.0
        tries = 0
        while True:
            raw = input('   move tool in %s, mm signed  (Enter when the jaws '
                        'straddle the cube) > ' % name).strip()
            if not raw:
                break
            try:
                d = float(raw) * 1e-3
            except ValueError:
                print('   not a number')
                continue
            if d == 0.0:
                break
            p = cell.tool_pose()
            tgt = [p[0], p[1]]
            tgt[axis] = p[axis] + d
            if not _move_tool(cell, session, tgt, p[2]):
                break
            total += d
            tries += 1
            print('   moved %+.0f mm in %s; total %+.0f mm from the first look'
                  % (d * 1e3, name, total * 1e3))

        # The tool ends on the cube, so the mount was short by -total along
        # that base axis. CAM_POS is in the flange frame: convert through the
        # flange rotation at the pose the measurement was taken at.
        _pf, R_flange = cell.kin().pose(cell.backend().joint_positions())
        delta = np.asarray(R_flange, float).T @ (np.eye(3)[axis] * total)
        session.cam_pos = [float(a + b) for a, b in
                           zip(session.cam_pos, delta)]
        print('   CAM_POS[%s] -> %.5f (%+.0f mm in the flange frame)'
              % (name, session.cam_pos[axis],
                 (session.cam_pos[axis] - start) * 1e3))
        out[name] = dict(residual_m=-total, tries=tries,
                         residual_history_m=[-total])
        if abs(total) <= OFFSET_TOL:
            print('   %s done: the mount was already right (total %+.1f mm)'
                  % (name, total * 1e3))
        else:
            print('   %s done: mount corrected by %+.1f mm of tool travel'
                  % (name, -total * 1e3))
    log['translation'] = out


def step_translation_check(cell, session, snaps, log):
    print()
    print('9  TRANSLATION CHECK -- repeat the centring at three positions')
    print('   No CAM_POS update. This only records the residual at each spot.')
    z = session.pick_z() + 0.150
    rows = []
    for label in ('centre', 'left', 'far'):
        input('   place the cube at the %s position, then Enter > ' % label)
        views = _look(cell, session, 'translation check %s' % label,
                      snaps, 'step9_%s' % label)
        v = _best(views, near=cell.tool_pose()[:2])
        if v is None:
            print('   no cube seen at the %s position -- skipped' % label)
            continue
        c = np.array(v.centre[:2], float)
        if not _move_tool(cell, session, c, z):
            continue
        ex = _ask_float('   residual x (mm, signed)', 0.0) * 1e-3
        ey = _ask_float('   residual y (mm, signed)', 0.0) * 1e-3
        r = float(np.hypot(ex, ey))
        rows.append(dict(position=label, residual_x_m=ex, residual_y_m=ey,
                         residual_m=r))
        print('   %-7s residual %.1f mm' % (label, r * 1e3))
    log['translation_check'] = rows
    if rows:
        worst = max(r['residual_m'] for r in rows)
        print('   worst residual %.1f mm' % (worst * 1e3))
        if worst > OFFSET_TOL:
            print('   FAIL: residual exceeds %.0f mm away from the first spot'
                  % (OFFSET_TOL * 1e3))
        else:
            print('   PASS: residual is consistent across positions')


def step_wrist_jog(cell, session, snaps, log):
    print()
    print('10 WRIST-JOG MOUNT CHECK -- static cube, rotate the wrist +-5 deg')
    print('   The cube centre should not move. A drift means CAM_QUAT is wrong.')
    input('   place the cube in view, then Enter > ')
    z = session.pick_z() + 0.150
    views = _look(cell, session, 'wrist jog base', snaps, 'step10_base')
    v = _best(views, near=cell.tool_pose()[:2])
    if v is None:
        print('   no cube seen -- skipped')
        return
    p = np.array([v.centre[0], v.centre[1], z], float)
    R0 = _flange_T(cell)[:3, :3]
    if not _move_pose(cell, p, R0):
        return
    views = _look(cell, session, 'wrist jog neutral', snaps, 'step10_neutral')
    v = _best(views, near=p[:2])
    if v is None:
        print('   no cube after the neutral move -- skipped')
        return
    base = np.array(v.centre[:2], float)

    from src.calibration_camera import rotvec_to_R
    rows = []
    tests = (('x+', [5.0, 0.0, 0.0]), ('x-', [-5.0, 0.0, 0.0]),
             ('y+', [0.0, 5.0, 0.0]), ('y-', [0.0, -5.0, 0.0]))
    try:
        for label, vec in tests:
            R = R0 @ rotvec_to_R(np.radians(np.asarray(vec, float)))
            if not _move_pose(cell, p, R):
                return
            views = _look(cell, session, 'wrist jog %s' % label,
                          snaps, 'step10_%s' % label.replace('+', 'p')
                          .replace('-', 'm'))
            v = _best(views, near=p[:2])
            if v is None:
                rows.append(dict(axis=label, drift_m=None))
                print('   %s: no cube in view' % label)
                continue
            c = np.array(v.centre[:2], float)
            drift = float(np.linalg.norm(c - base))
            rows.append(dict(axis=label, drift_m=drift,
                             centre_x_m=float(c[0]), centre_y_m=float(c[1])))
            print('   %s: centre (%+.4f, %+.4f)  drift %.1f mm'
                  % (label, c[0], c[1], drift * 1e3))
    finally:
        _move_pose(cell, p, R0)

    valid = [r['drift_m'] for r in rows if r['drift_m'] is not None]
    log['wrist_jog'] = rows
    if not valid:
        print('   FAIL: no usable looks')
        return
    worst = max(valid)
    print('   worst drift %.1f mm' % (worst * 1e3))
    if worst > 0.005:
        print('   FAIL: centre moves with the wrist; CAM_QUAT is wrong')
    else:
        print('   PASS: centre holds within 5 mm')


def step_mount_drift(cell, session, snaps, log):
    print()
    print('11 MOUNT DRIFT -- static cube, two poses 100 mm apart, no fit.')
    print('   Nothing is servoed and nothing is changed. If the reported centre')
    print('   moves with the tool, the mount rotation is wrong.')
    input('   place the cube in view, then Enter > ')
    z = session.pick_z() + 0.150
    p = cell.tool_pose()
    if not _move_tool(cell, session, p[:2], z):
        return
    views = _look(cell, session, 'drift start', snaps, 'step11_start')
    v = _best(views, near=cell.tool_pose()[:2])
    if v is None:
        print('   no cube in view -- nothing checked')
        return
    # Centre on the cube FIRST. With the tool already 100 mm off, the extra
    # 100 mm takes the cube out of the frame and the test records nothing.
    c0 = np.asarray(v.centre, float)
    if not _move_tool(cell, session, (c0[0], c0[1]), z):
        return
    views = _look(cell, session, 'drift A', snaps, 'step11_a')
    v = _best(views, near=cell.tool_pose()[:2])
    if v is None:
        print('   no cube after centring -- nothing checked')
        return
    a = np.array(v.centre[:2], float)
    moved = 0
    b = None
    for sign in (1.0, -1.0):
        p = cell.tool_pose()
        if not _move_tool(cell, session, (p[0] + sign * 0.10, p[1]), p[2]):
            continue
        views = _look(cell, session, 'drift %s' % ('B' if sign > 0 else 'C'),
                      snaps, 'step11_b' if sign > 0 else 'step11_c')
        v = _best(views, near=cell.tool_pose()[:2])
        if v is not None:
            b = np.array(v.centre[:2], float)
            moved = int(sign * 100)
            break
        print('   cube left the frame at %+d mm; trying the other way'
              % int(sign * 100))
    if b is None:
        print('   the cube left the frame in both directions -- nothing checked')
        return
    d = float(np.linalg.norm(b - a))
    print('   A (%+.4f, %+.4f)' % tuple(a))
    print('   B (%+.4f, %+.4f)   tool moved %+d mm in x'
          % (b[0], b[1], moved))
    print('   reported drift %.1f mm' % (d * 1e3))
    if d > 0.010:
        print('   FAIL: the report follows the arm -- mount rotation is wrong')
    else:
        print('   PASS: the report holds still')
    log['mount_drift'] = dict(a=[float(v) for v in a],
                              b=[float(v) for v in b], drift_m=d)


def step_angle(cell, session, snaps, log):
    print()
    print('4  ANGLE -- the cube is a fixed point in the BASE frame, so its place')
    print('   in the TOOL frame must not change. Look at it from several poses,')
    print('   rotating ONLY the wrist between them, and the tool fits the')
    print('   rotation that makes them agree.')
    print('   The translation is not fitted here; it comes from step 3.')
    input('   place the cube in view, then Enter > ')
    # 150 mm, the same look height step 3 uses. At 200 mm the flange pose is
    # 12 mm beyond the arm's reach from a cube at the front of the table
    # (elbow fully straight), and the step cannot start.
    z = session.pick_z() + 0.150
    p = cell.tool_pose()
    if not _move_tool(cell, session, p[:2], z):
        return
    from src.calibration_camera import rotvec_to_R

    pts = []
    tries = 0
    while tries < cfg.UR_TRACK_CHUNK:
        tries += 1
        views = _look(cell, session, 'step 4 pose %d' % (len(pts) + 1), snaps,
                      'step4_%02d' % (len(pts) + 1))
        v = _best(views, near=cell.tool_pose()[:2])
        if v is None:
            print('   no cube in view -- nudge the wrist and retry')
        else:
            hull = getattr(v, 'hull', None)
            if hull is None:
                print('   no top face hull on this look (the hardware view has '
                      'one; if not, the depth frame was thin) -- retry')
            else:
                T_tool = _flange_T(cell)
                p_cam = _point_in_camera(hull, cell, session)
                pts.append((T_tool, p_cam))
                print('   pose %d recorded (%d so far)' % (len(pts), len(pts)))
        raw = input('   [Enter] record another pose  [f] fit  [q] abandon > ')
        if raw.strip().lower() == 'q':
            return
        if raw.strip().lower() == 'f' or len(pts) >= 6:
            break
        # nudge: small xy move plus a wrist roll, so the cube stays central
        T_tool = _flange_T(cell)
        ang = np.radians(12.0)
        R_new = T_tool[:3, :3] @ rotvec_to_R(np.array([0.0, 0.0, ang]))
        # 5 mm per nudge, not 10: six poses at 10 mm walk the wrist 60 mm
        # further out, and the last two are past the reach limit at this
        # height. 5 mm keeps all six inside it.
        p = T_tool[:3, 3] + np.array([0.005, -0.005, 0.0])
        arm = cell.arm()
        q = arm.solve_ik(p, _seed(cell.backend().joint_positions()),
                         R_target=R_new, max_joint=2.0 * np.pi)
        arm.move(q, cfg.GRIP_OPEN, rate=cfg.FEED)

    if len(pts) < 3:
        print('   need at least 3 poses, got %d -- nothing fitted' % len(pts))
        return

    c_fit = np.asarray(session.cam_pos, float)
    R_cur = rot_from_quat(tuple(session.cam_quat))

    def make_R(ab):
        return R_cur @ rotvec_to_R(np.array([ab[0], ab[1], 0.0]))

    def residual(ab):
        R = make_R(ab)
        out = np.array([T[:3, 3] + T[:3, :3] @ (c_fit + R @ p_cam)
                        for T, p_cam in pts])
        # every pose must agree with the first
        return (out - out[0]).ravel()

    from scipy.optimize import least_squares
    print()
    print('   residual with the current quat      : %.2f mm'
          % (float(np.max(np.abs(residual([0.0, 0.0])))) * 1e3))
    sol = least_squares(residual, np.zeros(2))
    dev = float(np.max(np.abs(residual(sol.x)))) * 1e3
    R_fix = make_R(sol.x)
    from src.calibration_camera import quat_from_rot
    new_quat = quat_from_rot(R_fix)
    delta_deg = float(np.degrees(np.linalg.norm(sol.x)))
    print('   residual after fit                 : %.2f mm' % dev)
    print('   CAM_QUAT -> (%.6f, %.6f, %.6f, %.6f)' % tuple(new_quat))
    print('   correction %.2f deg tilt (roll about the optical axis held)'
          % delta_deg)
    if dev > 2.0:
        print('   ABOVE 2 mm: the poses disagree by more than one rotation can')
        print('   explain. The cube probably moved, or a pose was oblique.')
        print('   NOT applied -- re-run the step.')
        raw = input('   apply anyway? [y/N] > ').strip().lower()
        if not raw.startswith('y'):
            return
    session.cam_quat = [float(v) for v in new_quat]
    log['angle'] = dict(residual_mm=dev, poses=len(pts),
                        correction_deg=float(np.degrees(np.linalg.norm(sol.x))))


def _flange_T(cell):
    p, R = cell.kin().pose(cell.backend().joint_positions())
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    return T


def _point_in_camera(hull, cell, session):
    # The top face, averaged, expressed in the CAMERA frame.
    #
    # Uses the session mount only to undo the deprojection -- the hull arrived
    # in the base frame through that mount, so this is its camera-frame origin.
    # It is the one quantity that does not depend on the mount being right,
    # which is why the rotation is fitted from it.
    p, R = cell.world(session, cache=True).camera_pose()
    T_cam = np.eye(4)
    T_cam[:3, :3] = R
    T_cam[:3, 3] = p
    H = np.asarray(hull, float)
    world_pts = H
    P = (np.linalg.inv(T_cam) @ np.c_[world_pts, np.ones(len(world_pts))].T).T
    return P[:, :3].mean(axis=0)


def step_pick_height(cell, session, snaps, log):
    print()
    print('5  PICK HEIGHT -- the jaws close to the grip width and descend to the')
    print('   height you enter. TOOL height above the table, metres.')
    print('   MOVE THE CUBE OUT OF THE WAY first: you are judging the jaws.')
    session.pick_z_above_table = _ask_float(
        '  tool height above table, m', session.pick_z_above_table)
    # Close once, through the same call the pick makes, then descend holding it.
    cell.arm().close_gripper(cfg.STEPS_CLOSE)
    p = cell.tool_pose()
    z = session.pick_z()
    if not _move_tool(cell, session, p[:2], z, grip=cfg.GRIP_CLOSED,
                      rate=cfg.FEED / 2.0):
        return
    p = cell.tool_pose()
    print('  tool at z %.4f (table + %.1f mm)' % (p[2], (p[2] -
                                                       session.table_height) * 1e3))
    while not _ask_yes('  accept this pick height?', True):
        session.pick_z_above_table = _ask_float(
            '  tool height above table, m', session.pick_z_above_table)
        if not _move_tool(cell, session, cell.tool_pose()[:2],
                          session.pick_z(), grip=cfg.GRIP_CLOSED,
                          rate=cfg.FEED / 2.0):
            return
        p = cell.tool_pose()
        print('  tool at z %.4f' % p[2])
    log['pick_height'] = dict(above_table_m=session.pick_z_above_table,
                              tool_z=session.pick_z())


def step_grip_width(cell, session, snaps, log):
    print()
    print('6  GRIP WIDTH -- the jaws close to a width, lift %.0f mm and put the'
          % (GRIP_TEST_LIFT * 1e3))
    print('   cube back. If the cube comes up and the jaws stop ABOVE 50 mm,')
    print('   the width held. Force stays at %.1f N throughout.'
          % cfg.RG2_GRIP_FORCE_N)
    print('   The close is ONE rg_grip, the same call the pick makes.')
    arm = cell.arm()
    while True:
        width = _ask_float('  closing width, mm', session.close_width_mm)
        if width < cfg.RG2_MIN_STEP_MM:
            print('  below the %.1f mm minimum step' % cfg.RG2_MIN_STEP_MM)
            continue
        if width >= cfg.RG2_OPEN_WIDTH_MM:
            print('  must be narrower than the %.1f mm full stroke'
                  % cfg.RG2_OPEN_WIDTH_MM)
            continue
        pos = _grip_pos(width)
        _move_tool(cell, session, cell.tool_pose()[:2], session.pick_z(),
                   grip=cfg.GRIP_OPEN, rate=cfg.FEED / 2.0)
        input('  place the cube between the jaws, then Enter > ')

        # The same close the pick makes, but at the width under test rather
        # than the configured default.
        print('  closing to %.1f mm (command %.1f)' % (width, pos))
        result = arm.close_gripper(cfg.STEPS_CLOSE, position=pos)
        if result is None:
            print('  close_gripper gave no verdict -- is this a real gripper?')
            return
        ok, settled, verdict = result
        print('  closed: verdict %s, jaws stopped at %s mm'
              % (verdict, '?' if settled is None else '%.1f' % settled))

        cell_z = cell.tool_pose()[2]
        xy = cell.tool_pose()[:2]
        print('  lifting %.0f mm' % (GRIP_TEST_LIFT * 1e3))
        _move_tool(cell, session, xy, cell_z + GRIP_TEST_LIFT,
                   grip=pos, rate=cfg.FEED / 2.0)
        came_up = _ask_yes('  did the cube come up with the jaws?', True)
        print('  lowering and releasing')
        _move_tool(cell, session, xy, cell_z, grip=pos, rate=cfg.FEED / 2.0)
        arm.open_gripper(cfg.STEPS_RELEASE)

        if came_up:
            session.close_width_mm = width
            log['grip_width'] = dict(width_mm=width, pos=pos,
                                     force_n=cfg.RG2_GRIP_FORCE_N,
                                     settled_mm=settled, verdict=verdict,
                                     lifted=True)
            print('  RG2_CLOSE_WIDTH_MM -> %.1f' % width)
            if settled is not None and not ok:
                print('  NOTE: it lifted, but the settled width %.1f is not '
                      'above the 50 mm midpoint.' % settled)
                print('        The pick would still call this a FAIL. Fix the '
                      'midpoint or the close width.')
            return

        log.setdefault('grip_tries', []).append(
            dict(width_mm=width, settled_mm=settled, verdict=verdict,
                 lifted=False))
        print('  it did not hold -- try another width')


def step_readback(cell, session, snaps, log):
    print()
    print('7  READBACK -- what the robot will actually do')
    try:
        from rtde_receive import RTDEReceiveInterface
        r = RTDEReceiveInterface(cell.ip)
        tcp = r.getActualTCPPose()
        r.disconnect()
    except Exception as exc:
        print('  cannot read the TCP pose: %s' % exc)
        tcp = None
    p = cell.tool_pose()
    print('  FK flange pose      : (%.4f, %.4f, %.4f)' % tuple(p))
    if tcp is not None:
        print('  controller TCP pose : (%.4f, %.4f, %.4f)' % tuple(tcp[:3]))
        print('  |FK - TCP|          : %.1f mm'
              % (np.linalg.norm(np.asarray(tcp[:3]) - p) * 1e3))
    print('  commanded pick z    : %.4f (FK now %.4f)' % (session.pick_z(), p[2]))
    print('  HW_CAM_DELTA_*      : %s'
          % ('identity' if _same(cfg.HW_CAM_DELTA_POS, (0, 0, 0))
             else 'NON-IDENTITY'))
    log['readback'] = dict(fk=list(map(float, p)),
                           tcp=None if tcp is None else list(map(float, tcp)))


def step_empty_table(cell, session, snaps, log):
    print()
    print('8  EMPTY TABLE -- no cube. Anything detected here is a phantom or')
    print('   the mount seeing its own gripper.')
    input('   clear the table, then Enter > ')
    views = _look(cell, session, 'step 8 empty table', snaps, 'step8_empty')
    passed = not views
    print('  detections: %d' % len(views))
    log['empty_table'] = dict(detections=len(views), passed=passed)
    if passed:
        print('  PASS: an empty table finds nothing')
    else:
        print('  FAIL: PHANTOMS -- a look with nothing there must find nothing.')


# ------------------------------------------------------------------- save
def write_config(session):
    # Rewrite each constant in place by NAME, so formatting in config (how many
    # decimals a literal carries) cannot make a save silently miss.
    import re
    src = open(CONFIG_PATH).read()
    reps = [
        ('TABLE_HEIGHT', '%.6g' % session.table_height),
        ('CAM_POS', repr(tuple(round(v, 6) for v in session.cam_pos))),
        ('CAM_QUAT', repr(tuple(round(v, 6) for v in session.cam_quat))),
        ('PICK_Z_ABOVE_TABLE', '%.6g' % session.pick_z_above_table),
        ('RG2_CLOSE_WIDTH_MM', '%.6g' % session.close_width_mm),
    ]
    lines = []
    seen = {}
    for line in src.split('\n'):
        m = re.match(r'^([A-Z][A-Z0-9_]*) = ', line)
        if m and m.group(1) in dict(reps):
            name = m.group(1)
            # keep any trailing inline comment
            tail = ''
            if '#' in line:
                tail = '  ' + line[line.index('#'):]
            line = '%s = %s%s' % (name, dict(reps)[name], tail)
            seen[name] = True
        lines.append(line)
    missing = [n for n, _v in reps if n not in seen]
    if missing:
        print('  could not find %s in config -- NOT written'
              % ', '.join(missing))
        return False
    open(CONFIG_PATH, 'w').write('\n'.join(lines))
    print('  wrote %s' % CONFIG_PATH)
    return True


def write_scratch(session, log, snaps_dir):
    path = os.path.join(ROOT, 'calib_%s.json' % session.stamp)
    payload = dict(stamp=session.stamp,
                   values=dict(TABLE_HEIGHT=session.table_height,
                               CAM_POS=session.cam_pos,
                               CAM_QUAT=session.cam_quat,
                               PICK_Z_ABOVE_TABLE=session.pick_z_above_table,
                               PICK_Z=session.pick_z(),
                               RG2_CLOSE_WIDTH_MM=session.close_width_mm),
                   snaps=snaps_dir,
                   log=log)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print('  wrote %s' % path)


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog='calibrate_mount.py',
                                 description='Wrist mount and grasp calibration.')
    ap.add_argument('--ip', default=cfg.UR_IP)
    args = ap.parse_args(argv)
    start_run_log(argv if argv is not None else sys.argv[1:], ROOT,
                  tag='calibrate_mount')

    session = Session()
    snaps_dir = os.path.join(ROOT, 'calib_snaps_%s' % session.stamp)
    from src.liveview import Snapshotter
    snaps = Snapshotter(snaps_dir)
    cell = Cell(args.ip)
    log = {}

    try:
        while True:
            session.header()
            print('  1 table height   2 camera height   3 translation')
            print('  4 angle          5 pick height     6 grip width')
            print('  7 readback       8 empty-table check')
            print('  9 translation check  10 wrist-jog mount check')
            print('  11 mount drift check (static cube, 100 mm apart)')
            print('  h home')
            print('  s save to config  w write scratch JSON   r reset   q quit')
            choice = input('> ').strip().lower()
            if choice in ('q', 'quit'):
                break
            if choice == 's':
                if write_config(session):
                    write_scratch(session, log, snaps_dir)
                continue
            if choice == 'w':
                write_scratch(session, log, snaps_dir)
                continue
            if choice == 'r':
                keep = session.stamp
                session = Session()
                session.stamp = keep
                log = {}
                continue
            fn = {'h': step_home,
                  '1': step_table, '2': step_camera, '3': step_translation,
                  '4': step_angle, '5': step_pick_height,
                  '6': step_grip_width, '7': step_readback,
                  '8': step_empty_table, '9': step_translation_check,
                  '10': step_wrist_jog, '11': step_mount_drift}.get(choice)
            if fn is None:
                continue
            try:
                fn(cell, session, snaps, log)
            except KeyboardInterrupt:
                print('\n  step cancelled')
    finally:
        cell.close()
        if len(session.changed()):
            print('unsaved changes were discarded -- rerun with s or w')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
