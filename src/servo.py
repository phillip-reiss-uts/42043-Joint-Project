"""Position-Based Visual Servoing for the grasp approach.

HARDWARE BOUNDARY: this module is robot-side and contains no `import mujoco`.
It observes the world only through src.world.ObjectWorld (TargetView records),
gets geometry from arm.kin (src/kinematics.py) and steps through arm.backend
(src/robot_backend.py). It never imports the object pool or the cube driver.
"""

import numpy as np

from src.config import (
    YAW_TRACK_ENABLED,
    ARM_DOF, PICK_Z, R_DOWN, JOINT_RANGE, YAW_LIMIT_DEG,
    YAW_TRACK_DEADBAND_DEG, UR_SPEED_SCALE,
    GRIP_OPEN, SYNC_STEPS, RAPID, FEED,
    SERVO_USE_KF, KF_SIGMA_MEAS, KF_SIGMA_ACC, KF_GATE, SERVO_LEAD_S,
    SERVO_KP, SERVO_KO, SERVO_KI, SERVO_KI_NEAR_M, SERVO_KI_NEAR_RAD,
    SERVO_KI_MAX_M,
    SERVO_KI_MAX_RAD, SERVO_DAMP_GAIN, SERVO_DAMP_NEAR_M,
    SERVO_DAMP_NEAR_RAD,
    SERVO_TRACK_TOL, SERVO_TRACK_TOL_ROT, SERVO_STILL_SPEED, SERVO_LOST_LIMIT,
    SERVO_TRACK_MOVING, SERVO_STILL_TIME, SERVO_TRACK_ITERS, SERVO_STILL_TOL,
    SERVO_SEARCH_LIMIT, SERVO_YAW_SMOOTH, SERVO_WATCH_DESCENT,
    SERVO_MOVE_TOL, SERVO_COMMIT_TRIES,
    TABLE_HEIGHT,
    CONTACT_SLOW_ZONE, CONTACT_SLOW_FACTOR,
    UR_HOME_Q,
)
from src.tracking import TargetFilter
# Pure geometry helper; carries no object or simulator state.
from src.perception import rotation_with_yaw


def push(viewer, key, left, right=''):
    # Publish a one-line status, if a viewer with an overlay is attached.
    #
    # Local copy rather than `from src.overlay import push`, so this module does
    # not depend on the overlay layer: on hardware there is no viewer, and the
    # servo must compile without one. Duck-typed on purpose.
    ov = getattr(viewer, 'overlay', None)
    if ov is not None:
        ov.set_info(key, left, right)


# SERVOING
def _wrap90(d):
    # Shortest signed difference between two yaw readings that live modulo 90.
    return (np.asarray(d) + 45.0) % 90.0 - 45.0


def _tool_yaw(R):
    # The yaw the tool currently holds, degrees, from its orientation matrix.
    # A commanded orientation is Rz(yaw) @ R_DOWN, so R @ R_DOWN.T is Rz(yaw).
    Rz = np.asarray(R, float) @ R_DOWN.T
    return float(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0])))


def _bounded_yaw(yaw_ref, yaw_last, limit=YAW_LIMIT_DEG,
                 deadband=YAW_TRACK_DEADBAND_DEG):
    # The tool yaw to command for a face at yaw_ref, given the last commanded
    # yaw. A square face makes every yaw in {yaw_ref + 90k} an equally valid
    # grasp. Start from the member nearest the LAST COMMAND, then, if that is
    # outside +-limit, subtract/add 90 until it is back inside:
    #
    #     66  -> -24      100 -> 10      151 -> -29
    #
    # The result is always an EXACT cube face inside the limit, never a clamped
    # angle. Latching on the last command, not the measured tool yaw, stops the
    # 90 deg flip-flop at the boundary while the arm is still travelling.
    k = int(round((yaw_last - yaw_ref) / 90.0))
    cmd = float(yaw_ref + 90.0 * k)

    # DEADBAND. Do not re-command for caliper noise. Hold the last angle if the
    # exact face moved by only a couple of degrees. A real cube rotation beyond
    # the deadband still passes through to the subtract-90 correction below.
    if (abs(yaw_last) <= limit + 1e-9
            and abs(cmd - yaw_last) <= deadband):
        return float(yaw_last)

    while cmd > limit:
        cmd -= 90.0
    while cmd < -limit:
        cmd += 90.0
    return cmd


# Yaw disagreement above which a reading replaces the estimate instead of
# being blended into it, degrees.
YAW_JUMP_DEG = 20.0


class PBVSServo:
    # Drives the gripper to a pose derived from the current view of an object.

    def __init__(self, model, data, arm, world,
                 kp=SERVO_KP, ko=SERVO_KO, damp=1e-2, dq_max=0.02,
                 speed_scale=UR_SPEED_SCALE,
                 perceive_every=15, tol_p=3.0e-3, tol_o=3.0e-2,
                 smooth=1.0, v_max=float('inf'), v_descent=None,
                 yaw_smooth=SERVO_YAW_SMOOTH,
                 occlude_ratio=0.95, occlude_recover=0.97, occlude_fails=2,
                 retreat_step=0.02, retreat_max=15, retreat_v=None,
                 damp_gain=SERVO_DAMP_GAIN, damp_near_m=SERVO_DAMP_NEAR_M,
                 damp_near_rad=SERVO_DAMP_NEAR_RAD,
                 ki=SERVO_KI, ki_near_m=SERVO_KI_NEAR_M,
                 ki_near_rad=SERVO_KI_NEAR_RAD,
                 ki_max_m=SERVO_KI_MAX_M, ki_max_rad=SERVO_KI_MAX_RAD,
                 use_kf=SERVO_USE_KF, kf_sigma_meas=KF_SIGMA_MEAS,
                 kf_sigma_acc=KF_SIGMA_ACC, kf_gate=KF_GATE,
                 lead_s=SERVO_LEAD_S, track_tol=SERVO_TRACK_TOL,
                 track_tol_rot=SERVO_TRACK_TOL_ROT,
                 still_speed=SERVO_STILL_SPEED, lost_limit=SERVO_LOST_LIMIT,
                 still_tol=SERVO_STILL_TOL, search_limit=SERVO_SEARCH_LIMIT,
                 watch_descent=SERVO_WATCH_DESCENT, move_tol=SERVO_MOVE_TOL,
                 commit_tries=SERVO_COMMIT_TRIES,
                 track_moving=SERVO_TRACK_MOVING, still_time=SERVO_STILL_TIME,
                 track_iters=SERVO_TRACK_ITERS):
        self.model = model
        self.data = data
        self.arm = arm
        # The robot's only view of the world. Everything the servo knows about
        # the object arrives as a TargetView from here.
        self.world = world

        self.kp = kp                 # translation gain, 1/s
        self.ko = ko                 # rotation gain, 1/s
        self.damp = damp             # DLS damping
        self.damp_gain = float(damp_gain)      # joint-space D, near the target
        # Integral gain and its guards. ki=0 disables the I term entirely, which
        # is the default so the law stays purely proportional unless asked.
        self.ki = float(ki)
        self.ki_near_m = float(ki_near_m)      # only integrate inside this band
        self.ki_near_rad = float(ki_near_rad)
        self.ki_max_m = float(ki_max_m)        # clamp on the accumulated error
        self.ki_max_rad = float(ki_max_rad)
        self._err_int_p = np.zeros(3)
        self._err_int_o = np.zeros(3)
        # Commanded joint vector, persisted across servo() calls so that
        # chunking is equivalent to one long call. See the note in servo().
        self._q_cmd = None
        self.damp_near_m = float(damp_near_m)
        self.damp_near_rad = float(damp_near_rad)
        self.dq_max = dq_max         # joint increment clip per iteration (rad)
        # Hardware-calibrated global scale. 1.0 = uncalibrated command.
        self.speed_scale = float(np.clip(speed_scale, 0.01, 1.0))
        self.perceive_every = perceive_every
        self.tol_p = tol_p
        self.tol_o = tol_o
        # Blend factor applied to a new observation: 1.0 replaces the estimate
        # outright (a step change every perceive_every iterations, which the
        # arm answers with a lunge), lower values ease the target across.
        # Superseded by the Kalman filter when use_kf is set.
        self.smooth = float(np.clip(smooth, 0.0, 1.0))
        # Yaw gets its own gain, independent of the position blend. It lives
        # modulo 90 on a square top face, so it needs slower update than the
        # position or the label flips between equivalent readings.
        self.yaw_smooth = float(np.clip(yaw_smooth, 0.0, 1.0))
        # Vision Kalman filter: smooths the measurements, carries the estimate
        # through the stretches where the object is invisible, and supplies the
        # velocity for the lead term.
        self.use_kf = bool(use_kf)
        self.lead_s = float(lead_s)
        # tracking thresholds for the commit gate
        self.track_tol = float(track_tol)
        self.track_tol_rot = float(track_tol_rot)
        self.still_speed = float(still_speed)
        self.still_tol = float(still_tol)
        self.lost_limit = int(lost_limit)
        self.search_limit = int(search_limit)
        self.watch_descent = bool(watch_descent)
        self.move_tol = float(move_tol)
        self.commit_tries = int(commit_tries)
        # set when the camera sees the target move during an open-loop descent
        self.target_moved = False
        self.track_moving = bool(track_moving)
        self.still_time_target = float(still_time)
        self.track_iters = int(track_iters)
        self.kf = (TargetFilter(sigma_meas=kf_sigma_meas,
                                sigma_acc=kf_sigma_acc, gate=kf_gate)
                   if self.use_kf else None)
        # Commanded-twist caps (m/s) for the hover transit and the final
        # descent. inf reproduces the unbounded law.
        self.v_max = float(v_max)
        self.v_descent = (float(v_max) if v_descent is None
                          else float(v_descent))
        # Cartesian speed for the backup and the open-loop commit, so they move
        # like the servo rather than at a planned-move rate.
        self.retreat_v = (self.v_descent if retreat_v is None
                          else float(retreat_v))

        self.base_id = arm.base_id
        # Jacobians come from arm.kin (see src/kinematics.py); the servo no
        # longer needs its own MuJoCo scratch buffers.

        # held visual estimate, in world frame
        self.p_est = None
        self.yaw_est = None
        # Last wrist yaw COMMANDED, not the measured tool yaw. Used to latch
        # the equivalent-face branch across the +/-YAW_LIMIT boundary.
        self._yaw_cmd_last = None
        # last raw measurement, kept unfiltered so the harness can show what
        # the filter is actually removing
        self.raw_est = None
        self.last_dets = 0
        self.obs_ok = 0              # successful re-observations
        self.obs_try = 0             # attempts

        # Occlusion handling. The jaws clip the object on the way down, which
        # shrinks the measured span and pulls the fitted centre into the part
        # still visible. Track the fullest view and compare against it.
        self.occlude_ratio = float(occlude_ratio)
        self.occlude_recover = float(occlude_recover)
        self.occlude_fails = int(occlude_fails)
        self.retreat_step = float(retreat_step)
        self.retreat_max = int(retreat_max)

        self.span = None             # last measured (2,) xy extent
        self.span_ratio = 1.0        # vs the fullest view this stage
        self.best_span = 0.0         # largest span seen this stage
        self.best_p = None           # estimate taken at that fullest view
        self.best_yaw = None
        self.stage_obs = 0           # successful observations this stage
        self.fail_run = 0            # consecutive failed observations
        self.occluded = False        # occlusion onset seen on the descent
        self.recovered = False       # retreat + open-loop commit succeeded
        self.ref_p = None            # last unbiased estimate (fullest view)
        self.ref_yaw = None

        # Contact gate. Set when a commanded grasp pose would put the pads below
        # the tabletop; the pose is then clamped and the caller is told.
        self.gated = False
        self.gate_requested_z = None
        # Distance (m) between where a Cartesian descent was told to go and
        # where the tool actually ended. Non-zero means the arm did not follow
        # the line -- IK saturation or a stopped stream -- and closing the jaws
        # there would close them in mid-air.
        self.descent_short = 0.0
        self.descent_joint = None
        self.descent_info = None

        # set while the loop runs, so observe() can push to the overlay
        self.viewer = None

    def z_floor(self):
        # The z gate is the pick height itself: PICK_Z is the flange height
        # that places the pads on the object. No separate tool-length
        # clearance is applied.
        return PICK_Z

    def gate_report(self):
        # One-line description of a clamped grasp, for the terminal and overlay.
        return ('[SERVO] Z GATE: commanded TCP z %.3f is below the pick height '
                '%.3f -- stopped short by %.1f mm'
                % (self.gate_requested_z, self.z_floor(),
                   (self.z_floor() - self.gate_requested_z) * 1e3))

    def _control(self, state):
        # Tell the overlay which regime the loop is in, for colour coding.
        ov = getattr(self.viewer, 'overlay', None)
        if ov is not None:
            ov.set_control(state)

    def reset_command(self):
        # Forget the persisted joint command and the integral accumulator, so
        # the next servo() starts fresh from whatever the arm is doing now.
        # Call this when the servo is pointed at a DIFFERENT target; without it
        # a new grasp would inherit the command (and wind-up) left by the last.
        self._q_cmd = None
        self._yaw_cmd_last = None
        self._err_int_p = np.zeros(3)
        self._err_int_o = np.zeros(3)

    def _trace_camera(self):
        # Camera pose for the servo trace, if the world can supply one.
        pose = getattr(self.world, 'camera_pose', None)
        if pose is None:
            return {}
        p, R = pose()
        return {'cam_pos': np.array(p, float),
                'cam_mat': np.array(R, float).reshape(3, 3)}

    def _step_world(self):
        # Give the world one physics tick. Named so the intent is explicit: the
        # servo is not driving the object, it is advancing the simulation that
        # the object lives in. A hardware world would make this a no-op (the
        # real object moves on its own) or read a fresh sensor frame.
        step = getattr(self.world, 'step_mover', None)
        if step is not None:
            step()

    # PERCEPTION
    def _fold_yaw(self, new_yaw):
        # Blend a mod-90 caliper reading into the yaw estimate.
        #
        # The estimate is a CONTINUOUS angle that is only equivalent mod 90; it
        # is not clamped into [-45, 45). A large disagreement is taken whole
        # rather than blended.
        cand = float(_wrap90(new_yaw))
        if self.yaw_est is None:
            return cand
        step = float(_wrap90(cand - self.yaw_est))
        if abs(step) > YAW_JUMP_DEG:
            return float(self.yaw_est + step)
        return float(self.yaw_est + self.yaw_smooth * step)

    def observe(self, near=None):
        # Re-observe the object. Keeps the estimate nearest the previous one.
        #
        # Reads the world through the boundary only: whatever comes back is a
        # camera-derived TargetView, with no access to the object's true pose.
        self.obs_try += 1
        dets = self.world.perceive()
        self.last_dets = len(dets)
        ov = getattr(self.viewer, 'overlay', None)
        if ov is not None:
            ov.set_detections(dets)
        if not dets:
            push(self.viewer, 'perception',
                 'dets 0  re-obs %d/%d' % (self.obs_ok, self.obs_try),
                 'view blocked')
            return False
        if near is None:
            best = dets[0]
        else:
            best = min(dets, key=lambda d: np.linalg.norm(d.centre[:2] - near[:2]))
        new_est = np.array(best.centre, float)
        new_yaw = float(best.yaw)
        # A look with NO top-face rectangle has yaw = 0.0 as a placeholder, which
        # is not a measurement. Its CENTRE is still good, so the position
        # estimate takes it; only the yaw is held over from the last real one.
        yaw_ok = bool(getattr(best, 'yaw_ok', True))
        self.raw_est = new_est.copy()
        if self.kf is not None:
            # propagate to now, then fold the measurement in; a gated-out
            # measurement leaves the prediction standing
            self.kf.advance_to(float(self.arm.backend.time()))
            fresh = self.kf.update(new_est)
            self.p_est = self.kf.position
            if yaw_ok:
                self.yaw_est = self._fold_yaw(new_yaw)
        elif self.p_est is None:
            self.p_est = new_est
            if yaw_ok:
                self.yaw_est = self._fold_yaw(new_yaw)
        else:
            a = self.smooth
            self.p_est = self.p_est + a * (new_est - self.p_est)
            if yaw_ok:
                self.yaw_est = self._fold_yaw(new_yaw)
        if self.yaw_est is None:
            # Every look so far has had an unusable yaw. Fall back to the first
            # reading rather than to None, which nothing downstream expects.
            self.yaw_est = float(_wrap90(new_yaw))
        self.obs_ok += 1
        self.stage_obs += 1
        span = getattr(best, 'span', None)
        if span is not None and np.all(np.isfinite(span)):
            self.span = np.array(span, float)
            # the cube's AABB is near-isotropic, so the mean is a stable scalar
            span_norm = float(0.5 * (self.span[0] + self.span[1]))
            if span_norm > self.best_span:
                # fullest view so far: keep the estimate taken from it, because
                # it is the one the open-loop commit should fall back on
                self.best_span = span_norm
                self.best_p = np.array(self.p_est, float)
                self.best_yaw = float(self.yaw_est)
            self.span_ratio = (span_norm / self.best_span
                               if self.best_span > 0.0 else 1.0)
        push(self.viewer, 'perception',
             'dets %d  re-obs %d/%d' % (len(dets), self.obs_ok, self.obs_try),
             'span %.4f %.4f  %3.0f%% of best'
             % (self.span[0], self.span[1], 100.0 * self.span_ratio)
             if self.span is not None else 'span -')
        push(self.viewer, 'estimate',
             'est %6.3f %6.3f %6.3f' % tuple(self.p_est),
             'yaw %5.1f deg' % self.yaw_est)
        return True

    # GEOMETRY
    def _target_speed(self):
        # Estimated world speed of the target, for the stillness test. The
        # filter's velocity is the only sane source: a difference of raw
        # measurements is dominated by the ~1.5 mm of vision noise.
        if self.kf is not None and self.kf.ready:
            return self.kf.speed
        return None

    def desired_pose(self, z_tcp):
        # Tool0 pose that grasps the estimated object from above. With the
        # filter running this aims ahead of the estimate by lead_s, which
        # cancels the servo's own lag on a target that is still moving.
        p = self.p_est
        if self.kf is not None and self.kf.ready and self.lead_s != 0.0:
            p = self.kf.lead(self.lead_s)
        # z_tcp is an absolute tool0 height: the caller passes HOVER_TOOL0_Z
        # for hover and PICK_Z for the grasp. The estimate does not set it.
        z_want = float(z_tcp)
        # CONTACT GATE. PICK_Z is the only z floor: a command below it is
        # clamped, equal to it is the grasp. Remember that we clamped so the
        # callers report it and retreat rather than driving on.
        z_floor = self.z_floor()
        if z_want < z_floor:
            self.gated = True
            self.gate_requested_z = z_want
            z_want = z_floor
        p_des = np.array([p[0], p[1], z_want])
        # Yaw follows the CONTINUOUS estimate, not the fullest-view snapshot.
        # Holding best_yaw froze a cube that kept turning: yaw_est advanced but
        # the command did not. The estimate is already folded mod 90 and
        # smoothed by _fold_yaw/SERVO_YAW_SMOOTH, so it is the signal to track.
        # yaw_ref is the object's angle in the WORLD frame. Turning it into a
        # command picks which equivalent grasp to use, bounded by
        # YAW_LIMIT_DEG so the wrist cannot wind.
        yaw_ref = self.yaw_est if self.yaw_est is not None else self.best_yaw
        _p_cur, R_cur = self.current_pose()
        # Latch on the last COMMAND, not the measured tool yaw. The measured
        # yaw lags a 90 deg branch change, and using it lets boundary noise
        # flip the branch back before the arm gets there.
        if self._yaw_cmd_last is None:
            self._yaw_cmd_last = _tool_yaw(R_cur)
        if not YAW_TRACK_ENABLED or getattr(self, '_yaw_hold', False):
            # Hold the wrist where it is. Used by the open-loop commit: the
            # jaws are symmetric, so no wrist turn is needed at the grasp, and
            # a turn during the descent swings the tool sideways.
            yaw_cmd = _tool_yaw(R_cur)
        else:
            yaw_cmd = _bounded_yaw(yaw_ref, self._yaw_cmd_last)
        self._yaw_cmd_last = yaw_cmd
        R_des = rotation_with_yaw(yaw_cmd, R_DOWN)
        return p_des, R_des

    def current_pose(self):
        # The robot's own end-effector pose, via the boundary. Reading it from
        # the world rather than from `data` directly is what lets the same
        # servo run against a real arm, where there is no MuJoCo buffer.
        return self.world.gripper_pose()

    def error(self, p_des, R_des):
        # Pose error: translation vector and rotation vector, base frame.
        p_cur, R_cur = self.current_pose()
        err_p = p_des - p_cur
        eR = R_des @ R_cur.T
        err_o = np.array([eR[2, 1] - eR[1, 2],
                          eR[0, 2] - eR[2, 0],
                          eR[1, 0] - eR[0, 1]]) * 0.5
        return err_p, err_o

    # LAW
    def joint_step(self, err_p, err_o, v_max=None):
        # One damped-least-squares joint increment for this pose error.
        # The Jacobian comes from the arm's Kinematics provider, so this law is
        # the same on a simulated arm and a real one.
        J = self.arm.kin.jacobian(self.arm.backend.joint_positions())

        # INTEGRAL TERM. The law is P+I on the pose error; the I part is what
        # removes the standing sag.
        #
        # Why it is needed: the servo loop does `q_cmd += dq` every iteration. A
        # P-only law leaves a residual dq whenever the pose error is non-zero --
        # even a couple of millimetres -- and that residual ACCUMULATES into a
        # ramp. Measured at steady state on the UR3e: dq was 2.8e-4 per step, so
        # the commanded elbow walked 32 mrad downhill over 100 iterations and
        # qpos followed it. The arm visibly sags under that, the position servo
        # pulls back, and it rings -- the "sagging and catching itself" cycle.
        # The plant is not at fault: with ctrl frozen it holds to qvel_sd
        # 0.00000, and gravity feedforward is exact to 0.00 N.m.
        #
        # The integral is accumulated in the CARTESIAN error and re-expressed as
        # a twist, so it needs no separate Jacobian inverse and picks up the same
        # damping and limits as the proportional term.
        #
        # It lives on the instance, not in the loop: servo() is called in chunks
        # and re-seeds q_cmd from data.qpos each call, so a local accumulator
        # would reset every chunk and never build.
        if self.ki > 0.0:
            dt = self.arm.backend.dt
            self._err_int_p = self._err_int_p + np.asarray(err_p, float) * dt
            self._err_int_o = self._err_int_o + np.asarray(err_o, float) * dt

            # Conditional integration (anti-windup): freeze the accumulator while
            # the arm is far from the target or a limit is saturating. Without
            # this, a long transit or an occlusion -- during which the estimate
            # is frozen and the error therefore stays large -- winds the integral
            # up and produces a lurch when the target reappears.
            far = (np.linalg.norm(err_p) > self.ki_near_m
                   or np.linalg.norm(err_o) > self.ki_near_rad)
            if far:
                self._err_int_p = self._err_int_p - np.asarray(err_p, float) * dt
                self._err_int_o = self._err_int_o - np.asarray(err_o, float) * dt

            # Clamp: the integral can only ever contribute a bounded share of the
            # command, so even a pathology cannot make it dominate.
            for acc, lim in ((self._err_int_p, self.ki_max_m),
                             (self._err_int_o, self.ki_max_rad)):
                mag = float(np.linalg.norm(acc))
                if lim and mag > lim:
                    acc *= lim / mag

        v = np.concatenate([self.kp * err_p + self.ki * self._err_int_p,
                            self.ko * err_o + self.ki * self._err_int_o])

        # Cap the commanded twist. The law asks for kp*err outright, which is
        # metre-per-second scale at the start of an approach.
        #
        # BOTH halves must be capped. The original test measured the speed from
        # the POSITION error only (norm(v[:3])) and then scaled the whole
        # 6-vector by it. Such a guard can only ever scale DOWN, and only when
        # translation is too large, so the angular rate was never bounded at
        # all: with a small position error and a large yaw error it ran at
        # ko*0.8 = 4.8 rad/s unchecked. Near the hover that is exactly the
        # regime -- position converges while yaw has not -- so the arm swung the
        # tool up and over instead of settling, and ended with the gripper
        # pointing sideways at the sky rather than down at the table.
        #
        # The two parts are now limited separately. The rotational limit is
        # stated as the rate that moves the tool at v_max on a 0.15 m lever arm,
        # so a single v_max still means "how fast should this move" at both
        # scales, and neither limit acts unless exceeded.
        limit_v = (self.v_max if v_max is None else v_max) * self.speed_scale
        finite = np.isfinite(limit_v) and limit_v > 0.0

        n_t = float(np.linalg.norm(v[:3]))
        if finite and n_t > limit_v:
            v[:3] *= limit_v / n_t

        limit_w = (limit_v / 0.15) if finite else np.inf
        n_r = float(np.linalg.norm(v[3:]))
        if np.isfinite(limit_w) and n_r > limit_w:
            v[3:] *= limit_w / n_r

        dq = J.T @ np.linalg.solve(J @ J.T + self.damp * np.eye(6), v)
        dq = dq * self.arm.backend.dt

        # JOINT-SPACE DAMPING.
        #
        # The law above is purely kinematic: it converts a POSE error into a
        # joint increment with no reference to how fast the joints are already
        # moving. The position servos underneath have their own kv, but this
        # outer loop re-issues a command every iteration, so it keeps re-exciting
        # them. The result is a limit cycle rather than convergence: with the arm
        # correctly positioned (z error 0.4-2.3 mm) joint 2 still swung +/-0.35
        # rad/s and reversed sign between chunks. On screen that is the arm
        # sagging, catching itself, and sagging again instead of holding.
        #
        # Subtracting a term proportional to the measured joint velocity removes
        # the energy that drives that cycle. It is the D term the P law never
        # had, expressed in the same units as dq so it composes with everything
        # above: dq is a per-step increment, so the damping is a fraction of the
        # velocity's own per-step displacement.
        #
        # DAMP_GAIN = 1.0 cancels the joint velocity over one timestep, which is
        # critically damped for this discrete law. It is applied only when the
        # arm is already close to the target -- during a transit, velocity is
        # wanted, not damped.
        if self.damp_gain > 0.0:
            near = (np.linalg.norm(err_p) < self.damp_near_m
                    and np.linalg.norm(err_o) < self.damp_near_rad)
            if near:
                qvel = np.asarray(self.arm.backend.joint_velocities(), float)
                dq = dq - (self.damp_gain * qvel * self.arm.backend.dt)

        dq_max = self.dq_max * self.speed_scale
        return np.clip(dq, -dq_max, dq_max)

    def preposition(self, z_tcp, viewer=None):
        # Coarse open-loop move to the desired pose, before servoing.
        p_des, R_des = self.desired_pose(z_tcp)
        q = self.arm.solve_ik_tcp(p_des, self.arm.backend.joint_positions(), R_target=R_des)
        self.arm.move(q, GRIP_OPEN, rate=0.20, viewer=viewer)

    # LOOP
    def servo(self, z_tcp, max_iters=1500, viewer=None, estop=None,
              aborted=None, trace=None, v_max=None, check_occlusion=False,
              track=False, stationary_s=None, search=None,
              track_tol=None, track_tol_rot=None):
        # Servo to the pose at absolute tool0 height z_tcp.
        # Returns (converged, iters, err).
        backend = self.arm.backend
        self.viewer = viewer
        self._control('open')

        # each stage re-establishes its own fullest-view reference
        self.best_span = 0.0
        self.best_p = None
        self.best_yaw = None
        self.span_ratio = 1.0
        self.stage_obs = 0
        self.fail_run = 0
        self.occluded = False
        # tracking state: how long the target has looked still, and where we
        # last looked for it
        self.still_time = 0.0
        self.target_speed = None
        self.searched = 0
        self.fresh = False
        self.err_norm = float('inf')
        self.err_rot = float('inf')
        self._still_ref = None
        self._still_ref_t = float(backend.time())
        if track_tol is None:
            track_tol = self.track_tol
        if track_tol_rot is None:
            track_tol_rot = self.track_tol_rot

        if self.p_est is None and not self.observe():
            return False, 0, float('inf')

        # COMMANDED JOINTS PERSIST ACROSS CALLS.
        #
        # This used to re-seed q_cmd from data.qpos on every call, on the
        # reasoning that it made a short chunk "resume from where the arm is".
        # That is wrong, and it is what made tracking sag and jerk. Measured
        # with the same 600 physics steps either way:
        #
        #   servo(max_iters=600)      z 0.377 -> 0.376   err   1.76 mm
        #   600 x servo(max_iters=1)  z 0.377 -> 0.469   err 101.94 mm
        #
        # The loop is an integrator: it does q_cmd += dq each iteration, and a
        # P/integral law needs many iterations to close the error. Re-seeding
        # throws away the accumulated command every call, so a one-iteration
        # chunk commands a tiny step, the plant barely responds, and the next
        # call re-seeds from that barely-moved position and commands again. The
        # result is a command that ratchets downhill under gravity instead of
        # converging, which is exactly the sag-then-catch on screen.
        #
        # Keeping q_cmd on the instance makes repetition equivalent to one long
        # call, so the chunk size no longer changes the answer -- a caller can
        # run 1 iteration per frame or 600 and get the same motion.
        #
        # It is reset when the servo is pointed at a new target (see
        # _reset_command), so a genuinely new grasp does not inherit the old
        # command.
        if self._q_cmd is None or len(self._q_cmd) != ARM_DOF:
            self._q_cmd = np.array(backend.joint_positions(), float)
        q_cmd = np.array(self._q_cmd, float)
        err_norm = float('inf')
        last_t = float(backend.time())

        for k in range(max_iters):
            if aborted is not None and aborted():
                self._control('open')
                return False, k, err_norm

            dt = float(backend.time()) - last_t
            last_t = float(backend.time())

            if k % self.perceive_every == 0:
                # re-observe; hold the last estimate if the view is blocked
                fresh = self.observe(near=self.p_est)
                self.fresh = bool(fresh)
                # the overlay colours the frustum/trail by which regime we are in
                self._control('closed' if fresh else 'held')
                self.fail_run = 0 if fresh else self.fail_run + 1
                # Target gone from view completely: fall back to the search
                # routine to find it again, then carry on servoing. Only in
                # tracking mode -- on a servoed descent the object is meant to
                # be close, and losing it there is the occlusion case below.
                if (track and search is not None
                        and self.fail_run >= self.lost_limit):
                    self._control('open')
                    push(viewer, 'occlusion',
                         'TARGET LOST after %d looks' % self.fail_run,
                         'running search to re-acquire')
                    print('[SERVO] target lost at iter %d; searching' % k)
                    self.fail_run = 0
                    self.searched += 1
                    found = bool(search())
                    if found:
                        print('[SERVO] re-acquired, resuming servo')
                    else:
                        print('[SERVO] search found nothing')
                    last_t = float(backend.time())
                    # Give up rather than sweeping forever: the target is not on
                    # the table where we can see it, and spinning to the
                    # iteration cap only ties up the arm.
                    if not found and self.searched >= self.search_limit:
                        print('[SERVO] %d searches found nothing; giving up'
                              % self.searched)
                        self._control('open')
                        return False, k, err_norm
                    continue
                if check_occlusion and self.stage_obs >= 2:
                    # Two ways the jaws give themselves away: the measured span
                    # drops because the top face is being clipped, or the object
                    # disappears from view altogether. Either means the estimate
                    # is about to be the one we must not trust.
                    if (self.fail_run >= self.occlude_fails
                            or self.span_ratio < self.occlude_ratio):
                        self.occluded = True
                        self._control('held')
                        push(viewer, 'occlusion',
                             'OCCLUSION at iter %d: %s'
                             % (k, 'view lost %d obs' % self.fail_run
                                if self.fail_run >= self.occlude_fails
                                else 'span %3.0f%% of best'
                                     % (100.0 * self.span_ratio)),
                             'stopping and backing off')
                        return False, k, err_norm

            # carry the estimate forward to now, so the lead term is current
            # even across the stretches with no measurement
            if self.kf is not None and self.kf.ready:
                self.kf.advance_to(float(backend.time()))

            p_des, R_des = self.desired_pose(z_tcp)
            err_p, err_o = self.error(p_des, R_des)
            err_norm = float(np.linalg.norm(err_p))
            err_rot = float(np.linalg.norm(err_o))
            self.err_norm = err_norm
            self.err_rot = err_rot

            # Speed of the target, from the filter's velocity. Only observations
            # update it, so a still object reads still between looks rather than
            # decaying to zero on its own.
            self.target_speed = self._target_speed()
            if track and stationary_s:
                # Stillness measured as drift of the filtered position from an
                # anchor, not as the filter's velocity. The velocity carries a
                # few mm/s of noise against a 6 mm/s threshold, which kept
                # resetting the timer and delayed the commit by ~13 s.
                #
                # Only a fresh observation counts. A frozen estimate has zero
                # drift and would otherwise read as perfectly stationary, which
                # is exactly backwards: if the object is out of view we have no
                # evidence about it at all, and committing to the frozen
                # position grasps at empty table.
                if not self.fresh:
                    self.still_time = 0.0
                    self._still_ref = None
                else:
                    p_now = np.array(self.p_est, float)
                    if self._still_ref is None:
                        self._still_ref = p_now
                        self._still_ref_t = float(backend.time())
                    elif float(np.linalg.norm(p_now - self._still_ref)) \
                            > self.still_tol:
                        self._still_ref = p_now
                        self._still_ref_t = float(backend.time())
                    self.still_time = float(backend.time()) - self._still_ref_t

            ov = getattr(viewer, 'overlay', None)
            if ov is not None:
                ov.set_target(p_des, R_des)
            push(viewer, 'servo',
                 'z %.3f  iter %d/%d' % (z_tcp, k, max_iters),
                 'err %.4f m  tgt %s mm/s  still %.1f s'
                 % (err_norm,
                    '-' if self.target_speed is None
                    else '%5.1f' % (self.target_speed * 1e3),
                    self.still_time))

            if trace is not None:
                # Full record, so the loop can be replayed frame by frame.
                p_cur_t, _ = self.current_pose()
                trace.append({
                    'k': k,
                    'err_p': err_norm,
                    'err_o': err_rot,
                    'z_tcp': z_tcp,
                    'p_des': np.array(p_des, float),
                    'R_des': np.array(R_des, float),
                    'p_cur': np.array(p_cur_t, float),
                    'p_est': np.array(self.p_est, float),
                    'yaw': float(self.yaw_est),
                    # Camera pose for the trace. Taken from the world when it
                    # can supply one (SimObjectWorld has camera_pose();
                    # RealObjectWorld computes it from FK and the mount
                    # offset). Optional: a world without a camera simply omits
                    # these fields rather than breaking the loop.
                    **self._trace_camera(),
                })

            if track and stationary_s:
                # Commit gate for a target that was allowed to keep moving:
                # close enough to grasp, and confirmed still for long enough.
                # The strict tol_p/tol_o below would never be met while the
                # object is walking, so tracking uses its own looser band.
                if (self.still_time >= stationary_s
                        and err_norm < track_tol and err_rot < track_tol_rot):
                    self._control('open')
                    push(viewer, 'servo',
                         'STILL %.1f s at iter %d' % (self.still_time, k),
                         'committing open loop')
                    return True, k, err_norm
            elif err_norm < self.tol_p and err_rot < self.tol_o:
                self._control('open')
                return True, k, err_norm

            dq = self.joint_step(err_p, err_o, v_max=v_max)
            q_cmd = q_cmd + dq
            self._q_cmd = np.array(q_cmd, float)   # persists across calls
            # Command + tick through the arm. On the state-space path
            # command_step applies the observer/LQR correction; with no
            # observer attached it is the original raw command.
            self.arm.command_step(q_cmd, GRIP_OPEN,
                                  pre_step=self._step_world)

            while estop is not None and estop():
                self.arm.command_step(q_cmd, GRIP_OPEN)
                if viewer is not None:
                    viewer.sync()

            if viewer is not None and k % SYNC_STEPS == 0:
                viewer.sync()

        self._control('open')
        return False, max_iters, err_norm


    def _paced_move(self, q, d_cart, v_cart, viewer=None, estop=None,
                    aborted=None, watch=None):
        # arm.move paces on joint distance with a rad/s rate:
        #     steps = dist / rate / dt     (dist in joint rad)
        # so the duration is dist/rate, and the average Cartesian speed is
        # d_cart * rate / dist. Invert that to get the rate which yields the
        # speed the servo itself was running at. Exact for the average; the
        # move still eases in and out because arm.move always does.
        q_start = np.array(self.arm.backend.joint_positions(), float)
        dist = float(np.linalg.norm(np.asarray(q)[:ARM_DOF] - q_start))
        rate = RAPID
        if (v_cart is not None and np.isfinite(v_cart)
                and dist > 1e-9 and d_cart > 1e-9):
            rate = min(RAPID, dist * float(v_cart) / float(d_cart))

        # The target is driven by the world, not here. The per-step hook does
        # two things: tick the world's clock (so the cube keeps gliding during
        # this planned move, which is still physics), and run `watch` if given.
        def hook():
            self._step_world()
            if watch is not None:
                watch()
        self.arm.move(q, GRIP_OPEN, rate=rate, viewer=viewer, estop=estop,
                      stop=aborted, step_hook=hook)

    def _cartesian_descent(self, p_des, R_des, v_cart, viewer=None, estop=None,
                           aborted=None, watch=None):
        # Drive the tool in a straight line to p_des, holding R_des.
        #
        # The commit used to call arm.move() on a JOINT-space target: the
        # joints interpolate linearly, the tool does not, and the pads swing
        # out and back on the way down. Measured on the harness scenes: up to
        # 25 mm of lateral swing during the descent and 6-7 mm at the cube's
        # top face -- enough to put a pad against the cube before the jaws
        # close, which is what "the gripper drifts forward and hits it" is.
        #
        # Here the TOOL path is the interpolated quantity and the joints are
        # solved per step, so the pads come down the line the camera aimed at.
        # The IK is warm-started from the previous solution, so it costs a few
        # iterations per step rather than a full solve.
        backend = self.arm.backend
        p0, _ = self.world.gripper_pose()
        p0 = np.asarray(p0, float)
        p_des = np.asarray(p_des, float)
        # Slow the final stretch when this run ENDS near the contact gate. The
        # quintic profile already eases out, but a descent that finishes within a
        # few millimetres of the table should arrive gently as well.
        if p_des[2] - self.z_floor() <= CONTACT_SLOW_ZONE:
            v_cart = max(float(v_cart) * CONTACT_SLOW_FACTOR, 1e-3)
        dist = float(np.linalg.norm(p_des - p0))
        steps = max(1, int(dist / max(float(v_cart), 1e-6) / backend.dt))
        q = np.array(backend.joint_positions(), float)
        self.descent_short = 0.0
        self.descent_joint = None
        self.descent_info = None

        for i in range(steps):
            if aborted is not None and aborted():
                return
            a = (i + 1) / steps
            a = a * a * a * (a * (a * 6.0 - 15.0) + 10.0)
            p = (1.0 - a) * p0 + a * p_des
            # Per-joint limits, not one scalar: the solver default of +-3 rad
            # is a simulator convention, and a wrist that needs past it gets
            # clipped. The tool then follows a pose that is NOT on the line and
            # the descent finishes high -- which closed the jaws in mid-air.
            # The elbow is the one joint the MJCF models at +-pi; the rest are
            # +-2*pi, as on the UR3e.
            q = self.arm.solve_ik_tcp(p, q, R_target=R_des, max_joint=JOINT_RANGE)
            backend.command(q, GRIP_OPEN)
            self._step_world()
            if watch is not None and i % self.perceive_every == 0:
                watch()
            backend.step()
            while estop is not None and estop():
                backend.command(q, GRIP_OPEN)
                backend.step()
                if viewer is not None:
                    viewer.sync()
            if viewer is not None and i % SYNC_STEPS == 0:
                viewer.sync()

        p_end, _ = self.world.gripper_pose()
        self.descent_short = float(np.linalg.norm(
            np.asarray(p_end, float) - p_des))
        # Why it fell short: the IK's own residual (the commanded joint vector
        # does not put the tool on p_des) versus the joint lag (the arm is not
        # where it was commanded). Those need different fixes.
        p_cmd, _ = self.arm.kin.pose(q[:ARM_DOF])
        q_act = np.asarray(backend.joint_positions(), float)[:ARM_DOF]
        self.descent_info = {
            'steps': steps,
            'ik_mm': 1e3 * float(np.linalg.norm(np.asarray(p_cmd) - p_des)),
            'lag': float(np.max(np.abs(q[:ARM_DOF] - q_act))),
        }
        # Which joint is closest to its clip, so a short descent can say why.
        lim = np.abs(np.asarray(JOINT_RANGE, float))
        if lim.ndim:
            margin = lim - np.abs(q[:len(lim)])
            self.descent_joint = (int(np.argmin(margin)),
                                  float(q[int(np.argmin(margin))]),
                                  float(lim[int(np.argmin(margin))]))

    def retreat(self, viewer=None, estop=None, aborted=None, v_cart=None):
        # Back straight up in steps until the object is whole in view again.
        # Each step is a planned move, so the retreat itself is open loop; only
        # the verdict comes from the camera. Returns (clear, steps).
        if v_cart is None:
            v_cart = self.retreat_v
        p_cur, R_cur = self.current_pose()
        anchor = self.best_p if self.best_p is not None else self.p_est
        for j in range(1, self.retreat_max + 1):
            if aborted is not None and aborted():
                return False, j - 1
            target = p_cur + np.array([0.0, 0.0, self.retreat_step * j])
            q = self.arm.solve_ik_tcp(target, self.arm.backend.joint_positions(),
                                     R_target=R_cur)
            self._paced_move(q, self.retreat_step, v_cart, viewer=viewer,
                             estop=estop, aborted=aborted)
            if aborted is not None and aborted():
                return False, j
            if (self.observe(near=anchor)
                    and self.span_ratio >= self.occlude_recover):
                push(viewer, 'occlusion',
                     'retreated %.0f mm over %d steps'
                     % (self.retreat_step * j * 1e3, j),
                     'span back to %3.0f%% of best' % (100.0 * self.span_ratio))
                return True, j
        push(viewer, 'occlusion',
             'retreat exhausted %d steps' % self.retreat_max, 'still clipped')
        return False, self.retreat_max

    def motion_watch(self, viewer=None):
        # Called during the open-loop descent. Camera only -- no ground truth,
        # nothing from the object's own driver. Re-observe and ask whether the
        # target has walked away from the position we committed to.
        #
        # The span guard is what makes this usable: once the jaws start clipping
        # the object the fitted centre is biased by up to 0.55x any lateral
        # offset, and that bias grows as the gripper descends. Without it the
        # robot would read its own approach as the object moving. Past the point
        # where the view is no longer whole we simply stop judging.
        if not self.observe(near=self.ref_p):
            return
        if self.span_ratio < self.occlude_ratio:
            return
        d = float(np.linalg.norm(np.array(self.p_est, float) - self.ref_p))
        if d > self.move_tol:
            self.target_moved = True
            print('[SERVO] target moving again: %.0f mm from the commit '
                  'reference; abandoning the open-loop descent' % (d * 1e3))
            push(viewer, 'occlusion', 'TARGET MOVING AGAIN',
                 '%.0f mm from commit ref -- abandoning descent' % (d * 1e3))

    def commit_open_loop(self, viewer=None, estop=None, aborted=None,
                         watch=None):
        # Drive to the grasp pose from the last unbiased estimate with no further
        # correction, so the jaws cannot re-bias the target on the way in.
        ref = self.ref_p if self.ref_p is not None else self.best_p
        if ref is None:
            return False, 0
        self.p_est = np.array(ref, float)
        self.yaw_est = float(self.ref_yaw if self.ref_yaw is not None
                             else self.best_yaw)
        # RE-LATCH THE YAW BRANCH TO THE WRIST THAT IS ACTUALLY THERE.
        # The jaws are symmetric, so yaw and yaw+-180 are the same grasp -- but
        # they are a 180 deg wrist turn, and this commit is open loop: the spin
        # happens during the descent and swings the tool ~25 mm sideways, which
        # is what made the descent stop 28 mm short with the right height.
        # Choosing the equivalent face nearest the CURRENT wrist turns at most
        # 45 deg, normally not at all.
        _p_now, R_now = self.current_pose()
        self._yaw_cmd_last = _tool_yaw(R_now)
        # One height for the grasp and the descent it replaces: PICK_Z, applied
        # in desired_pose. The contact gate still clamps it to the pad limit, so
        # it cannot drive into the table. The yaw hold is only for THIS target:
        # it is cleared immediately after, so later looks track normally.
        self._yaw_hold = True
        try:
            p_des, R_des = self.desired_pose(PICK_Z)
        finally:
            self._yaw_hold = False
        ov = getattr(self.viewer, 'overlay', None)
        if ov is not None:
            ov.set_target(p_des, R_des)
        push(viewer, 'occlusion',
             'open-loop commit to %6.3f %6.3f' % (self.best_p[0], self.best_p[1]),
             'from the fullest view, span %.4f' % self.best_span)

        # OPEN LOOP, which is what the commit is for: one IK solve from the last
        # unbiased estimate, then a plain joint move. No per-step Cartesian
        # servo, no re-solving down the line, no camera in the loop. The reason
        # the camera must stay out is the same one that makes the servo descent
        # unusable here -- closer in, the jaws and the housing bias the fitted
        # centre, so anything that keeps looking drives on a drifting target.
        self.target_moved = False
        q_goal = self.arm.solve_ik_tcp(
            p_des, np.asarray(self.arm.backend.joint_positions(), float),
            R_target=R_des, max_joint=JOINT_RANGE)
        # What the solver actually produced. A joint move ends at FK(q_goal),
        # so a horizontal miss here is the SOLVER's residual, not the move's:
        # the arm cannot land anywhere else than q_goal.
        p_ik, _ = self.arm.kin.pose(np.asarray(q_goal[:ARM_DOF], float))
        ik_mm = float(np.linalg.norm(np.asarray(p_ik, float)
                                    - np.asarray(p_des, float)) * 1e3)
        if ik_mm > 5.0:
            # The solver ran out of iterations or into a clip from this seed.
            # One retry from the home configuration, which is a known-good
            # branch, before giving up: a 28 mm horizontal miss on the cell was
            # the solver landing short, not the arm failing to move.
            q_alt = self.arm.solve_ik_tcp(
                p_des, np.asarray(UR_HOME_Q, float),
                R_target=R_des, max_joint=JOINT_RANGE)
            p_alt, _ = self.arm.kin.pose(np.asarray(q_alt[:ARM_DOF], float))
            alt_mm = float(np.linalg.norm(np.asarray(p_alt, float)
                                          - np.asarray(p_des, float)) * 1e3)
            print('[IK] seed gave %.1f mm, home seed gives %.1f mm'
                  % (ik_mm, alt_mm))
            if alt_mm < ik_mm:
                q_goal, ik_mm = q_alt, alt_mm
        self.descent_info = dict(ik_mm=ik_mm, lag=0.0, steps=0)
        self.arm.move(q_goal, GRIP_OPEN, rate=FEED, estop=estop)
        # SETTLE. move() returns when the last target has been STREAMED, not
        # when the arm has arrived; servoJ lags it. Measuring straight away
        # reads that lag as a short descent (13 mm on the cell), so hold the
        # final target for a moment before asking where the tool is.
        for _ in range(max(1, int(1.0 / self.arm.backend.dt))):
            self.arm.command_step(q_goal, GRIP_OPEN)
        self.descent_short = 0.0
        p_end, _ = self.world.gripper_pose()
        self.descent_short = float(np.linalg.norm(
            np.asarray(p_end, float) - np.asarray(p_des, float)))
        if self.gated:
            # The pose we wanted was below the contact limit, so the descent
            # stopped short of it. Say so, back off, and report failure rather
            # than closing the jaws in mid-air over a table we never reached.
            msg = self.gate_report()
            print(msg)
            push(viewer, 'occlusion', 'Z GATE: grasp not reached', msg)
            self.retreat(viewer=viewer, estop=estop, aborted=aborted)
            return False, 0
        if self.target_moved:
            # The monitor saw the TARGET move during the descent. That is a
            # different failure from an unreachable pose and used to be hidden
            # behind the short-descent message below.
            push(viewer, 'occlusion', 'descent abandoned: target moved',
                 '%.0f mm from the grasp pose' % (1e3 * self.descent_short))
            print('[SERVO] DESCENT ABANDONED: the target moved while descending '
                  '(%.0f mm short, not closing)' % (1e3 * self.descent_short))
            return False, 0
        # DO NOT CLOSE HIGH. If the tool did not reach the grasp pose, report a
        # failed descent so the caller can retry; closing here is what grabbed
        # the cube with the pads above it.
        if self.descent_short > 0.010:
            extra = ''
            if self.descent_joint is not None:
                j, qv, ql = self.descent_joint
                extra = ('  [j%d at %+.2f, clip %+.2f]'
                         % (j + 1, qv, ql if qv >= 0 else -ql))
            info = getattr(self, 'descent_info', None) or {}
            why = ''
            if info:
                why = ('  [ik %.0f mm, joint lag %.2f rad, %d steps]'
                       % (info['ik_mm'], info['lag'], info['steps']))
            msg = ('[SERVO] DESCENT SHORT: stopped %.0f mm above the grasp pose '
                   '(wanted z %.3f at (%+.3f, %+.3f), reached z %.3f at '
                   '(%+.3f, %+.3f)) -- not closing%s%s'
                   % (1e3 * self.descent_short,
                      p_des[2], p_des[0], p_des[1],
                      p_end[2], p_end[0], p_end[1], extra, why))
            print(msg)
            push(viewer, 'occlusion', 'descent short -- not closing', msg)
            return False, 0
        if aborted is not None and aborted():
            return False, 0
        return True, 1

    def approach(self, hover_z, viewer=None, estop=None, aborted=None,
                 trace=None, coarse=True, max_iters=1500, recover=True,
                 check_occlusion=True, track=None, search=None,
                 stationary_s=None, track_iters=None):
        # Coarse move to the hover pose, then closed-loop hover and descent.
        if self.p_est is None and not self.observe():
            return False, 0, float('inf'), 0
        self.gated = False
        self.gate_requested_z = None
        if track is None:
            # A stationary target is picked with the direct hover/descent
            # path. Tracking is for a target that is actually moving; waiting
            # for stillness on a noisy stationary estimate can loop forever.
            speed = self._target_speed()
            track = bool(self.track_moving and speed is not None
                         and speed > SERVO_STILL_SPEED)
        if stationary_s is None:
            stationary_s = self.still_time_target

        if track:
            # Keep servoing for as long as the object keeps moving. The stage
            # ends when the target has been watched still long enough, at which
            # point the grasp is driven open loop from the hover -- where the
            # view is still whole -- instead of servoing down into the jaws'
            # own shadow.
            if coarse:
                self.preposition(hover_z, viewer=viewer)
            n_done = 0
            for attempt in range(max(1, self.commit_tries)):
                ok1, n1, e1 = self.servo(
                    hover_z, max_iters=(track_iters or self.track_iters),
                    viewer=viewer, estop=estop, aborted=aborted, trace=trace,
                    v_max=self.v_max, track=True,
                    stationary_s=stationary_s, search=search)
                n_done += n1
                if not ok1:
                    return False, n_done, e1, n_done
                self.observe(near=self.p_est)
                self.ref_p = np.array(self.p_est, float)
                self.ref_yaw = float(self.yaw_est)
                ok_c, n_c = self.commit_open_loop(
                    viewer=viewer, estop=estop, aborted=aborted,
                    watch=self.watch_descent)
                n_done += n_c
                if ok_c:
                    self.recovered = True
                    return True, n_done, 0.0, n_done
                if self.gated:
                    # commit_open_loop already printed the gate message and
                    # retreated. Do not retry: the target did not move, the pose
                    # is simply not reachable without hitting the table.
                    return False, n_done, e1, n_done
                if not self.target_moved:
                    return False, n_done, e1, n_done
                # The camera caught the target moving part way down. Lift back
                # to the hover on the next pass and chase it again; the servo
                # drives to the hover pose, so it climbs out on its own.
                push(viewer, 'occlusion', 'ABANDONED descent %d' % (attempt + 1),
                     'target moved -- resuming the chase')
            push(viewer, 'occlusion', 'gave up after %d descents'
                 % self.commit_tries, 'target would not hold still')
            return False, n_done, e1, n_done

        if coarse:
            self.preposition(hover_z, viewer=viewer)
        ok1, n1, e1 = self.servo(hover_z, max_iters=max_iters, viewer=viewer,
                                 estop=estop, aborted=aborted, trace=trace,
                                 v_max=self.v_max)
        if not ok1:
            return False, n1, e1, 0
        # Reference estimate for the open-loop commit. Taken here, while the
        # object is still whole in view: closer in, the jaws bias the fitted
        # centre and the estimate feeds back into the gripper position, so this
        # is the last unbiased view on the way down.
        self.observe(near=self.p_est)
        self.ref_p = np.array(self.p_est, float)
        self.ref_yaw = float(self.yaw_est)
        ok2, n2, e2 = self.servo(PICK_Z, max_iters=max_iters, viewer=viewer,
                                 estop=estop, aborted=aborted, trace=trace,
                                 v_max=self.v_descent,
                                 check_occlusion=check_occlusion)
        if self.gated:
            # The grasp pose we were asked for is below the contact limit. The
            # servo has driven to the clamped height instead, so the pick was NOT
            # reached: say so, back off, and fail.
            msg = self.gate_report()
            print(msg)
            push(viewer, 'occlusion', 'Z GATE: grasp not reached', msg)
            self.retreat(viewer=viewer, estop=estop, aborted=aborted)
            return False, n1 + n2, e2, n1
        if ok2:
            return True, n1 + n2, e2, n1
        if not (recover and self.occluded):
            return False, n1 + n2, e2, n1
        # the jaws clipped the object on the way down: back off until the view
        # is whole, then commit the grasp open loop from that estimate
        ok_r, n_r = self.retreat(viewer=viewer, estop=estop, aborted=aborted)
        if not ok_r:
            return False, n1 + n2 + n_r, e2, n1
        ok_c, n_c = self.commit_open_loop(viewer=viewer, estop=estop,
                                          aborted=aborted)
        if ok_c:
            self.recovered = True
            return True, n1 + n2 + n_r, 0.0, n1
        return False, n1 + n2 + n_r + n_c, e2, n1
