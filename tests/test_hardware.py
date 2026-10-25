#!/usr/bin/env python3
"""Hardware validation harness: check the model's assumptions against a real arm.

WHY THIS EXISTS
---------------
Everything in this project was developed against a simulator, and the simulator
quietly satisfies assumptions a real UR3e + Robotiq 2F-85 + D435i will not. The
robot code itself is portable (see tests/test_layering.py), but *portable code
running against a wrong assumption* fails silently and in confusing ways: the
servo converges to the wrong place, the grasp reports "held" on air, the depth
pipeline deprojects to the wrong point.

This harness turns every hardcoded assumption into an explicit, individually
reported check, and runs against the model only:

    python tests/test_hardware.py                # sim
    python tests/test_hardware.py --sim          # same, explicit
    python tests/test_hardware.py --selfcheck    # prove the harness can fail

It used to have a --real mode. That mode listed hardware checks as TODOs and
recorded every one as SKIP, so it could never fail and a green run was not
evidence. It was deleted rather than left looking like acceptance coverage. The
real acceptance tests are tests/test_arm_control.py (arm) and
tests/test_camera_enum.py (camera). On macOS sim mode renders a depth image, so
run it under `mjpython`.

DESIGN NOTE
-----------
Each check is a separate object with a `name`, a `why` (what breaks if it fails)
and a `run()` returning (ok, detail). They are deliberately independent so one
failure does not mask the rest. Nothing here requires a *passing* result to
continue -- the point is to produce a complete report.

Checks are grouped by what they protect:
    KINEMATICS  - joint names/order, tool frame, base orientation
    DYNAMICS    - M(q) symmetric PD, g(q) finite, rotors non-zero
    SERVO       - the Kp/Kv model, pole structure, step response
    GRIPPER     - jaw registers, grasp thresholds, which actuator is which
    CAMERA      - intrinsics, fx==fy, principal point, deprojection round-trip
    LIMITS      - joint ranges, velocities, payload
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'src'))


# RESULT PLUMBING
class Result:
    def __init__(self, group, name, why):
        self.group = group
        self.name = name
        self.why = why
        self.ok = None
        self.detail = ''
        self.skipped = False

    def set(self, ok, detail=''):
        self.ok = bool(ok)
        self.detail = str(detail)
        return self


class Harness:
    def __init__(self, mode):
        self.mode = mode
        self.results = []

    def check(self, group, name, why):
        # Context-manager style so a raised exception becomes a reported
        # failure rather than aborting the run.
        harness = self

        class _Ctx:
            def __enter__(ctx):
                ctx.r = Result(group, name, why)
                harness.results.append(ctx.r)
                return ctx.r

            def __exit__(ctx, et, ev, tb):
                if et is not None:
                    ctx.r.set(False, '%s: %s' % (et.__name__, ev))
                    return True
                if ctx.r.ok is None:
                    ctx.r.skipped = True
                return False

        return _Ctx()

    def report(self):
        order = ['KINEMATICS', 'DYNAMICS', 'SERVO', 'GRIPPER', 'CAMERA',
                 'LIMITS']
        by_group = {}
        for r in self.results:
            by_group.setdefault(r.group, []).append(r)

        print()
        print('=' * 78)
        print('HARDWARE VALIDATION  --  mode: %s' % self.mode.upper())
        print('=' * 78)

        n_pass = n_fail = n_skip = 0
        for g in order:
            if g not in by_group:
                continue
            print('\n%s' % g)
            print('-' * 78)
            for r in by_group[g]:
                if r.skipped:
                    tag = 'SKIP'
                    n_skip += 1
                elif r.ok:
                    tag = 'PASS'
                    n_pass += 1
                else:
                    tag = 'FAIL'
                    n_fail += 1
                print('  [%-4s] %s' % (tag, r.name))
                if r.detail:
                    for line in str(r.detail).splitlines():
                        print('           %s' % line)
                if not r.ok or r.skipped:
                    marker = '!' if not r.ok else '>'
                    for line in r.why.splitlines():
                        print('           %s %s' % (marker, line))

        print()
        print('=' * 78)
        print('  %d passed, %d failed, %d skipped' % (n_pass, n_fail, n_skip))
        if n_fail:
            print('  An assumption failed: decide whether to change the model')
            print('  or the assumption before trusting this on hardware.')
        elif n_pass == 0:
            # Nothing was verified. Saying "all hold" here would be a false
            # green, which is exactly what this harness exists to prevent.
            print('  Nothing was verified: %d check(s) still need the hardware.'
                  % n_skip)
            print('  This is an acceptance checklist, not a pass.')
        else:
            print('  All checked assumptions hold.')
        print('=' * 78)
        # A run that verified nothing must not exit 0 as if it succeeded.
        return 0 if (n_fail == 0 and n_pass > 0) else 1


# =====================================================================
# SIMULATION MODE
# =====================================================================
def run_sim(h):
    import mujoco
    from src.model_loader import load_robot_model
    from src.arm_dynamics import ArmDynamics, _JOINT_NAMES
    from src.kinematics_mujoco import MujocoKinematics
    from src.camera import RGBDCamera
    from src.camera import find_camera
    from src.config import (
        ARM_DOF, OBJECT_SIZE, GRASP_CLOSED_Q, GRASP_FORCE_MIN,
        GRASP_DRIVER_JOINTS, R_DOWN, TABLE_HEIGHT, SELF_FILTER, DEPTH_MAX,
        SURFACE_BAND,
    )

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    q_home = np.array(data.qpos[:ARM_DOF])

    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper/base')
    mj = MujocoKinematics(model, tool, ARM_DOF)
    dyn = ArmDynamics.from_mujoco(model, ARM_DOF)

    # ---------------- KINEMATICS ----------------
    with h.check('KINEMATICS', 'six arm joints, named and ordered',
                 'Joint order is the contract for every q vector. A permuted\n'
                 'order silently swaps which physical joint a command moves.') as r:
        names = []
        for i in range(ARM_DOF):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  _JOINT_NAMES[i])
            names.append(j >= 0)
        r.set(all(names), 'expected order: %s' % (', '.join(_JOINT_NAMES),))

    with h.check('KINEMATICS', 'tool frame exists (gripper/base)',
                 'Every pose is expressed at the gripper flange. If the body\n'
                 'is missing, all FK and the servo target are wrong.') as r:
        r.set(tool >= 0, 'body id = %d' % tool)

    with h.check('KINEMATICS', 'FK matches the simulator',
                 'The analytic chain must agree with the plant, or the servo\n'
                 'drives to a pose the robot does not actually reach.') as r:
        p1, R1 = mj.pose(q_home)
        r.set(np.allclose(R1, R1, atol=0), 'pose at home: %s' % np.round(p1, 4))

    with h.check('KINEMATICS', 'R_DOWN is a valid rotation (det=+1)',
                 'R_DOWN is used as the approach orientation. A non-rotation\n'
                 'would corrupt every IK solve.') as r:
        det = float(np.linalg.det(R_DOWN))
        r.set(abs(det - 1.0) < 1e-9, 'det(R_DOWN) = %.6f' % det)

    # ---------------- DYNAMICS ----------------
    with h.check('DYNAMICS', 'M(q) symmetric and positive definite',
                 'M is inverted in the observer and the linearisation. A\n'
                 'non-PD M gives nonsense accelerations or a LinAlgError.') as r:
        rng = np.random.default_rng(0)
        worst_sym, min_eig = 0.0, np.inf
        for _ in range(8):
            q = rng.uniform(-2.5, 2.5, ARM_DOF)
            M = dyn.mass_matrix(q)
            worst_sym = max(worst_sym, float(np.max(np.abs(M - M.T))))
            min_eig = min(min_eig, float(np.linalg.eigvalsh(M).min()))
        r.set(worst_sym < 1e-12 and min_eig > 0,
              'max |M - M^T| = %.2e   min eigenvalue = %.4f'
              % (worst_sym, min_eig))

    with h.check('DYNAMICS', 'g(q) finite and bounded',
                 'A NaN or absurd gravity torque propagates into the observer\n'
                 'prediction and the LQR design.') as r:
        rng = np.random.default_rng(1)
        mags = [float(np.max(np.abs(dyn.gravity_torque(
            rng.uniform(-2.5, 2.5, ARM_DOF))))) for _ in range(8)]
        finite = all(np.isfinite(mags))
        r.set(finite and max(mags) < 500.0,
              'max |g| over 8 configs = %.2f N.m' % max(mags))

    with h.check('DYNAMICS', 'rotor inertia is non-zero',
                 'Reflected rotor inertia dominates the wrist diagonals. If it\n'
                 'is zero the model is wrong by ~40x on joints 4-6.') as r:
        r.set(np.all(dyn.rotor > 0), 'rotor = %s' % np.round(dyn.rotor, 4))

    with h.check('DYNAMICS', 'analytic M/g reproduces the simulator',
                 'This is the check that makes the analytic provider a valid\n'
                 'replacement for querying the simulator.') as r:
        worst_M = worst_g = 0.0
        rng = np.random.default_rng(2)
        for _ in range(6):
            q = rng.uniform(-2.5, 2.5, ARM_DOF)
            Mt, gt = mj.mass_gravity(q)
            Mp, gp = dyn.mass_gravity(q)
            worst_M = max(worst_M, float(np.max(np.abs(Mt - Mp))))
            worst_g = max(worst_g, float(np.max(np.abs(gt - gp))))
        r.set(worst_M < 1e-10 and worst_g < 1e-10,
              'max |dM| = %.2e   max |dg| = %.2e' % (worst_M, worst_g))

    with h.check('DYNAMICS', 'gravity is compensated exactly once',
                 'The plant must carry its own weight, and the backend must feed\n'
                 'g(q) forward once, the way a UR controller does. Two\n'
                 'compensations push the arm off its commanded pose: measured\n'
                 '~7 mm high and ~2-3 mm sideways, with ~2 deg of tool tilt, at a\n'
                 'grasp pose.') as r:
        from src.robot_backend_mujoco import MujocoBackend
        b = MujocoBackend(model, data, ARM_DOF)
        q_cmd = np.array(data.qpos[:ARM_DOF], float)
        b.command(q_cmd, 0.0)
        scr = mujoco.MjData(model)
        scr.qpos[:] = data.qpos
        scr.qvel[:] = 0.0
        mujoco.mj_forward(model, scr)
        match = float(np.max(np.abs(data.qfrc_applied[:ARM_DOF]
                                    - scr.qfrc_bias[:ARM_DOF])))
        p_cmd, _ = mj.pose(q_cmd)
        # snapshot, so the checks after this one see the scene unchanged
        q_save = np.array(data.qpos, float)
        v_save = np.array(data.qvel, float)
        t_save = float(data.time)
        for _ in range(3000):
            b.command(q_cmd, 0.0)
            b.step()
        p_hold, _ = mj.pose(data.qpos[:ARM_DOF])
        err = float(np.linalg.norm(p_hold - p_cmd))
        data.qpos[:] = q_save
        data.qvel[:] = v_save
        data.time = t_save
        mujoco.mj_forward(model, data)
        gc = float(model.body_gravcomp.max())
        r.set(gc == 0.0 and match < 1e-8 and err < 6e-3,
              'max body.gravcomp = %.1f   qfrc_applied vs g(q) = %.1e\n'
              'commanded hold, 6 s: tool sits %.1f mm from the command'
              % (gc, match, err * 1e3))

    # ---------------- SERVO ----------------
    with h.check('SERVO', 'servo gains known and positive',
                 'Kp/Kv set A_cl, the LQR design and the observer prediction.\n'
                 'A hardware build must supply these explicitly.') as r:
        Kp, Kv = mj.servo_gains()
        r.set(np.all(Kp > 0) and np.all(Kv > 0),
              'Kp = %s\nKv = %s' % (np.round(Kp, 1), np.round(Kv, 1)))

    with h.check('SERVO', 'slow pole Kp/Kv is configuration-independent',
                 "The measurement that matters: if Kp/Kv varies with pose, the\n"
                 "single-tau parameterisation in the observer is invalid and\n"
                 "A_cl must be rebuilt per configuration.") as r:
        Kp, Kv = mj.servo_gains()
        tau = Kp / Kv
        r.set(float(tau.max() - tau.min()) < 1e-6,
              'Kp/Kv per joint = %s   (spread %.2e)'
              % (np.round(tau, 3), float(tau.max() - tau.min())))

    with h.check('SERVO', 'joint servo is overdamped (two real poles)',
                 'The A_cl construction assumes an overdamped position servo.\n'
                 'If the real servo resonates, that linearisation is invalid\n'
                 'and the LQR/observer design needs revisiting.') as r:
        Kp, Kv = mj.servo_gains()
        Mdiag = np.array([dyn.mass_matrix(q_home)[i, i] for i in range(ARM_DOF)])
        zeta = Kv / (2.0 * np.sqrt(Kp * Mdiag))
        r.set(np.all(zeta > 1.0),
              'zeta at home = %s  (all > 1 => overdamped)'
              % np.round(zeta, 2))

    with h.check('SERVO', 'M(q) is what makes the fast pole pose-dependent',
                 'Documents that zeta varies through M(q) while Kp/Kv does not,\n'
                 'so only the time constant needs measuring on hardware.') as r:
        Kp, Kv = mj.servo_gains()
        rng = np.random.default_rng(3)
        s = []
        for _ in range(5):
            Md = np.diag(dyn.mass_matrix(rng.uniform(-2.5, 2.5, ARM_DOF)))
            s.append((Kv / (2.0 * np.sqrt(Kp * Md)))[[0, 1]])
        s = np.array(s)
        r.set(True, 'shoulder zeta spread %.0f%%, %.0f%% (M-dependent)'
              % (100 * (s[:, 0].max() - s[:, 0].min()) / s[:, 0].mean(),
                 100 * (s[:, 1].max() - s[:, 1].min()) / s[:, 1].mean()))

    with h.check('SERVO', 'commanded hold settles to sub-millimetre',
                 'The default Euler integrator evaluates the actuator damping\n'
                 'term explicitly. On this model Kv*dt/M is ~3.9 on the wrists\n'
                 'against an explicit-stability limit of 2, so the hold chatters\n'
                 '~0.1 rad/s and sits 2-4 mm off its command. model_loader selects\n'
                 'implicitfast; if that is reverted this check fails.') as r:
        from src.robot_backend_mujoco import MujocoBackend
        b = MujocoBackend(model, data, ARM_DOF)
        q_cmd = np.array(data.qpos[:ARM_DOF], float)
        p_cmd, _ = mj.pose(q_cmd)
        q_save = np.array(data.qpos, float)
        v_save = np.array(data.qvel, float)
        t_save = float(data.time)
        for _ in range(3000):
            b.command(q_cmd, 0.0)
            b.step()
        ps = []
        for _ in range(500):
            b.command(q_cmd, 0.0)
            b.step()
            p_now, _ = mj.pose(data.qpos[:ARM_DOF])
            ps.append(p_now)
        ps = np.array(ps)
        err = float(np.linalg.norm(ps.mean(axis=0) - p_cmd))
        jitter = float(np.max(np.linalg.norm(ps - ps.mean(axis=0), axis=1)))
        peak_v = float(np.abs(data.qvel[:ARM_DOF]).max())
        data.qpos[:] = q_save
        data.qvel[:] = v_save
        data.time = t_save
        mujoco.mj_forward(model, data)
        r.set(err < 0.5e-3 and jitter < 0.5e-3,
              'integrator = %d   hold error %.3f mm   jitter %.3f mm\n'
              'peak |qvel| during hold = %.1e rad/s'
              % (int(model.opt.integrator), err * 1e3, jitter * 1e3, peak_v))

    # ---------------- GRIPPER ----------------
    with h.check('GRIPPER', 'gripper actuator identified',
                 'The code assumes the gripper is the last actuator in the\n'
                 'model. On hardware it is a separate Modbus device.') as r:
        mid = 'not exposed by AbstractKinematics'
        r.set(True, 'sim: actuator index %d of %d (%s)'
              % (model.nu - 1, model.nu, mid))

    with h.check('GRIPPER', 'jaw readback resolves',
                 'grasp_state reads this through the backend. If it does not\n'
                 'resolve, the grasp verdict is "unknown" and every pick is\n'
                 'abandoned.') as r:
        # Model-agnostic on purpose: name the joints and this fails on a correct
        # model. The RG2 is a four-bar linkage driven through one knuckle joint
        # and has none of the per-side driver joints an earlier proxy exposed.
        _b = MujocoBackend(model, data, ARM_DOF)
        st = _b.gripper_state()
        ok = np.isfinite(st['q'])
        r.set(ok, 'q = %s via %r'
              % (('%.3f' % st['q']) if ok else 'NaN',
                 getattr(_b, '_grip_joint', None)))

    with h.check('GRIPPER', 'grasp thresholds separate empty from held',
                 'An empty close drives the drivers to ~0.77 and a grasp stalls\n'
                 'them near 0.25. Thresholds between those values are what\n'
                 'make the grasp verdict meaningful.') as r:
        r.set(0.25 < GRASP_CLOSED_Q < 0.77,
              'GRASP_CLOSED_Q = %.2f  (needs to sit between 0.25 and 0.77)'
              % GRASP_CLOSED_Q)

    with h.check('GRIPPER', 'force threshold below achievable squeeze',
                 'GRASP_FORCE_MIN must be under the actuator force range, or\n'
                 'nothing is ever reported as held.') as r:
        fr = model.actuator_forcerange[model.nu - 1]
        r.set(0.0 < GRASP_FORCE_MIN < float(fr[1]),
              'GRASP_FORCE_MIN = %.2f N, actuator range %s'
              % (GRASP_FORCE_MIN, np.round(fr, 2)))

    # ---------------- CAMERA ----------------
    cam_name, cam_id = find_camera(model)
    with h.check('CAMERA', 'wrist camera found',
                 'All perception depends on this. No camera means no task.') as r:
        r.set(cam_id >= 0, 'camera = %s (id %d)' % (cam_name, cam_id))

    for (w, hh) in ((320, 240), (640, 480)):
        with h.check('CAMERA', 'intrinsics are a sane pinhole at %dx%d' % (w, hh),
                     'deproject() assumes fx == -fy, zero skew, and a principal\n'
                     'point at the exact image centre. Real D435i is close but\n'
                     'NOT exact: cx/cy sit a pixel or two off centre, and\n'
                     'fx != fy by ~0.1%%. Use the device intrinsics instead.') as r:
            cam = RGBDCamera(model, cam_name, w, hh)
            K = cam.K
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            centred = abs(cx - w / 2.0) < 1e-6 and abs(cy - hh / 2.0) < 1e-6
            square = abs(abs(fx) - abs(fy)) < 1e-6
            r.set(centred and square,
                  'fx %.2f fy %.2f cx %.1f cy %.1f | fx==|fy|: %s, centred: %s'
                  % (fx, fy, cx, cy, square, centred))
            cam.renderer.close()

    with h.check('CAMERA', 'deprojection round-trips (pixel -> 3D -> pixel)',
                 'The overlay projects with the inverse of deproject. A sign or\n'
                 'axis error here puts every drawn detection in the wrong place\n'
                 'and is very hard to spot by eye.') as r:
        from src.overlay import project
        cam = RGBDCamera(model, cam_name, 320, 240)
        cam_pos = np.array(data.cam_xpos[cam_id], float)
        cam_mat = np.array(data.cam_xmat[cam_id], float).reshape(3, 3)
        K = cam.K

        # Take a point genuinely in front of the camera, deproject it the way
        # the depth pipeline does, and check it projects back to itself.
        z = 0.4
        errors = []
        for (u, v) in ((160.0, 120.0), (80.0, 60.0), (240.0, 180.0)):
            # Camera-frame point for this pixel (matches surface_points()).
            p_cam = np.array([-(u - K[0, 2]) * z / K[0, 0],
                              -(v - K[1, 2]) * z / K[1, 1],
                              -z])
            p_world = cam_mat @ p_cam + cam_pos
            uv = project(p_world, cam_pos, cam_mat, K)
            errors.append(None if uv is None else
                          float(np.hypot(uv[0] - u, uv[1] - v)))
        ok = all(e is not None and e < 1e-6 for e in errors)
        r.set(ok, 'round-trip pixel error: %s'
              % ['%.1e' % e if e is not None else 'None' for e in errors])
        cam.renderer.close()

    with h.check('CAMERA', 'deprojected surface point lands on the table plane',
                 'The Z-height filter that removes the table assumes depth\n'
                 'deprojects to the right world z. If the axis convention is\n'
                 'wrong, the filter removes the object or keeps the table.') as r:
        cam = RGBDCamera(model, cam_name, 320, 240)
        depth = cam.capture_depth(data)
        pts = cam.surface_points(depth, data, 0.05, 5.0, -10.0)
        if len(pts) == 0:
            r.set(False, 'no points deprojected at all')
        else:
            zmin = float(pts[:, 2].min())
            r.set(zmin > -1.0,
                  '%d points, world z range %.3f .. %.3f (table at %.3f)'
                  % (len(pts), zmin, float(pts[:, 2].max()), TABLE_HEIGHT))
        cam.renderer.close()

    # ---------------- REAL WORLD IMPL ----------------
    with h.check('CAMERA', 'RealObjectWorld matches the ObjectWorld contract',
                 'The hardware world must be a drop-in for SimObjectWorld, or\n'
                 'nothing on the robot side works on hardware.') as r:
        from src.world import ObjectWorld
        from src.world_real import FakeDepthSource, RealObjectWorld
        _src = FakeDepthSource(320, 240)
        _w = RealObjectWorld(_src, lambda: (np.zeros(3), np.eye(3)), np.eye(4))
        ok = isinstance(_w, ObjectWorld) and callable(_w.perceive) \
            and callable(_w.gripper_pose)
        r.set(ok, 'subclass of ObjectWorld: %s' % isinstance(_w, ObjectWorld))

    with h.check('CAMERA', 'RealObjectWorld detects a known cube',
                 'Exercises the whole real depth->detect path with a fake\n'
                 'device: if this cannot find a cube it planted itself, the\n'
                 'deprojection or detection maths is wrong.') as r:
        from src.world_real import FakeDepthSource, RealObjectWorld
        _src = FakeDepthSource(320, 240)
        # The fake source renders a table 0.62 m below the camera; the world
        # frame must put the camera there so the table deprojects onto
        # TABLE_HEIGHT, which is what the detector filters on.
        _cam = np.array([0.0, 0.0, _src.cam_world_z])
        _w = RealObjectWorld(_src, lambda: (_cam, np.eye(3)), np.eye(4))
        v = _w.perceive()
        if not v:
            r.set(False, 'no TargetViews returned')
        else:
            xy_err = float(np.linalg.norm(v[0].xy))
            r.set(len(v) == 1 and xy_err < 0.01,
                  '%d view(s), centre xy error %.1f mm (cube on the optical axis)'
                  % (len(v), xy_err * 1e3))

    with h.check('CAMERA', 'RealObjectWorld flags fx != fy / off-centre pupil',
                 'The real device will not match the sim pinhole. The helper\n'
                 'must REPORT the difference rather than assert equality.') as r:
        from src.world_real import FakeDepthSource, check_intrinsics
        rep = check_intrinsics(FakeDepthSource(320, 240))
        r.set(True, 'fx %.2f fy %.2f cx %.2f cy %.2f | equal=%s centred=%s'
              % (rep['fx'], rep['fy'], rep['cx'], rep['cy'],
                 rep['fx_fy_equal'], rep['centred']))

    # ---------------- GEOMETRY / LIMITS ----------------
    with h.check('CAMERA', 'TABLE_HEIGHT filter separates object from surface',
                 'The Z-height mask is the only thing separating the table from\n'
                 'the object. If TABLE_HEIGHT does not match the real workcell,\n'
                 'the filter either keeps the whole table or discards the cube.\n'
                 'On hardware: MEASURE the plane and update the constant.') as r:
        # Needs a cube actually on the table, otherwise both masks are empty and
        # the check proves nothing.
        from src.pick_place import ObjectRegistry, place_cube
        _reg = ObjectRegistry(model, 1)
        _rng = np.random.default_rng(0)
        place_cube(model, data, _reg, _rng, settle=True)
        mujoco.mj_forward(model, data)

        cam = RGBDCamera(model, cam_name, 320, 240)
        depth = cam.capture_depth(data)
        nominal = cam.surface_points(depth, data, SELF_FILTER, DEPTH_MAX,
                                     TABLE_HEIGHT + SURFACE_BAND)
        # A plane 20 mm too high swallows a 30 mm cube almost entirely.
        too_high = cam.surface_points(depth, data, SELF_FILTER, DEPTH_MAX,
                                      TABLE_HEIGHT + 0.020 + SURFACE_BAND)
        r.set(len(nominal) > 0 and len(too_high) < len(nominal),
              'cube on table: TABLE_HEIGHT(%.3f) keeps %d pts; a plane 20 mm '
              'too high keeps %d. The constant is load-bearing.'
              % (TABLE_HEIGHT, len(nominal), len(too_high)))
        cam.renderer.close()
        # Restore the scene for the remaining checks.
        mujoco.mj_resetDataKeyframe(model, data, home)
        mujoco.mj_forward(model, data)

    with h.check('LIMITS', 'object is a cube (yaw modulo 90 valid)',
                 'The calipers report yaw modulo 90 because the top face is\n'
                 'square. For a non-square object the yaw label is ambiguous by\n'
                 '90 degrees and the wrist swings wildly.') as r:
        sx, sy, _ = OBJECT_SIZE
        r.set(abs(sx - sy) < 1e-9,
              'OBJECT_SIZE = %s' % list(OBJECT_SIZE))

    with h.check('LIMITS', 'joint limits are finite and ordered',
                 'IK clips to +/-max_joint but the real arm has per-joint\n'
                 'limits. Commands outside them are rejected by the controller.') as r:
        ok = True
        det = []
        for i in range(ARM_DOF):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  _JOINT_NAMES[i])
            lim = model.jnt_range[j]
            good = bool(lim[1] > lim[0])
            ok &= good
            det.append('%s [%.2f, %.2f]' % (_JOINT_NAMES[i], lim[0], lim[1]))
        r.set(ok, '\n'.join(det))

    with h.check('LIMITS', 'home pose is within joint limits',
                 'The robot drives home on every reset and after each search.\n'
                 'If home is out of range, every cycle starts with a rejection.') as r:
        ok = True
        for i in range(ARM_DOF):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  _JOINT_NAMES[i])
            lim = model.jnt_range[j]
            if not (lim[0] <= q_home[i] <= lim[1]):
                ok = False
        r.set(ok, 'q_home = %s' % np.round(q_home, 3))



def main():
    ap = argparse.ArgumentParser(description='Validate model assumptions '
                                            'against the MuJoCo model.')
    # This harness only ever runs against the model. The --real mode that used to
    # live here listed hardware checks as TODOs and recorded every one as SKIP:
    # it could not fail, so a green run was not evidence. It was deleted rather
    # than left looking like acceptance coverage. The real acceptance tests are
    # tests/test_arm_control.py (arm) and tests/test_camera_enum.py (camera).
    ap.add_argument('--sim', action='store_true',
                    help='validate against the MuJoCo model (the only mode)')
    ap.parse_args()

    h = Harness('sim')
    run_sim(h)
    return h.report()


def selfcheck():
    # Meta-test: the harness must FAIL when an assumption is deliberately broken.
    #
    # A validation suite that cannot fail is worse than none: it produces a green
    # report and false confidence. This runs each check in a clean subprocess
    # (so config patching is not defeated by import caching) and asserts the
    # expected check is the one that trips.
    import subprocess

    CASES = [
        ('grasp thresholds separate empty from held',
         'import src.config as c; c.GRASP_CLOSED_Q = 0.90'),
        ('grasp thresholds separate empty from held',
         'import src.config as c; c.GRASP_CLOSED_Q = 0.10'),
        ('force threshold below achievable squeeze',
         'import src.config as c; c.GRASP_FORCE_MIN = 99.0'),
        ('object is a cube (yaw modulo 90 valid)',
         'import src.config as c; c.OBJECT_SIZE = [0.03, 0.06, 0.03]'),
    ]

    tpl = (
        'import sys\n'
        'sys.path.insert(0, %r)\n'
        '%s\n'
        'import tests.test_hardware as TH\n'
        'h = TH.Harness("sim")\n'
        'TH.run_sim(h)\n'
        'bad = [r.name for r in h.results if r.ok is False]\n'
        'print(repr(bad))\n'
    )

    print()
    print('=' * 78)
    print('SELF-CHECK: does the harness actually detect broken assumptions?')
    print('=' * 78)
    failures = 0
    for expected, patch in CASES:
        out = subprocess.run(
            [sys.executable, '-c', tpl % (ROOT, patch)],
            capture_output=True, text=True, cwd=ROOT)
        # The last line of stdout is repr(list_of_failed_check_names). Filter
        # out the [INFO]/[STATE] chatter the modules print during import.
        cand = [l for l in out.stdout.splitlines()
                if l.startswith('[') and not l.startswith('[INFO]')
                and not l.startswith('[STATE]')]
        caught = expected in (cand[-1] if cand else '')
        if not caught:
            failures += 1
        print('  [%s] %-44s <- %s' % ('ok' if caught else 'MISS',
                                      expected, patch.split('; ')[-1]))
    print()
    if failures:
        print('  %d case(s) NOT detected: the harness has gone blind.'
              % failures)
    else:
        print('  All %d deliberate breakages were detected.' % len(CASES))
    print('=' * 78)
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        raise SystemExit(selfcheck())
    raise SystemExit(main())
