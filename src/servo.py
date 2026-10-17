# Position-Based Visual Servoing for the grasp approach.
#
# HARDWARE BOUNDARY: this module is robot-side and contains no `import mujoco`.
# It observes the world only through src.world.ObjectWorld (TargetView records),
# gets geometry from arm.kin (src/kinematics.py) and steps through arm.backend
# (src/robot_backend.py). It never imports the object pool or the cube driver.

import numpy as np

from src.config import (
    ARM_DOF, TIP_OFFSET, R_DOWN, TABLE_HEIGHT, OBJECT_HEIGHT,
    GRIP_OPEN, GRIP_CLOSED, SYNC_STEPS, RAPID,
    SERVO_USE_KF, KF_SIGMA_MEAS, KF_SIGMA_ACC, KF_GATE, SERVO_LEAD_S,
    SERVO_TRACK_TOL, SERVO_TRACK_TOL_ROT, SERVO_STILL_SPEED, SERVO_LOST_LIMIT,
    SERVO_TRACK_MOVING, SERVO_STILL_TIME, SERVO_TRACK_ITERS, SERVO_STILL_TOL,
    SERVO_SEARCH_LIMIT, SERVO_YAW_SMOOTH, SERVO_WATCH_DESCENT,
    SERVO_MOVE_TOL, SERVO_COMMIT_TRIES,
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


class PBVSServo:
    # Drives the gripper to a pose derived from the current view of an object.

    def __init__(self, model, data, arm, world,
                 kp=6.0, ko=6.0, damp=1e-2, dq_max=0.02,
                 perceive_every=15, tol_p=3.0e-3, tol_o=3.0e-2,
                 smooth=1.0, v_max=float('inf'), v_descent=None,
                 yaw_smooth=SERVO_YAW_SMOOTH,
                 occlude_ratio=0.95, occlude_recover=0.97, occlude_fails=2,
                 retreat_step=0.02, retreat_max=15, retreat_v=None,
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
        self.dq_max = dq_max         # joint increment clip per iteration (rad)
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

        # set while the loop runs, so observe() can push to the overlay
        self.viewer = None

    def _control(self, state):
        # Tell the overlay which regime the loop is in, for colour coding.
        ov = getattr(self.viewer, 'overlay', None)
        if ov is not None:
            ov.set_control(state)

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
        self.raw_est = new_est.copy()
        if self.kf is not None:
            # propagate to now, then fold the measurement in; a gated-out
            # measurement leaves the prediction standing
            self.kf.advance_to(float(self.data.time))
            fresh = self.kf.update(new_est)
            self.p_est = self.kf.position
            self.yaw_est = (new_yaw if self.yaw_est is None else
                            float((self.yaw_est
                                   + self.yaw_smooth * _wrap90(new_yaw
                                                               - self.yaw_est))
                                  % 90.0))
        elif self.p_est is None:
            self.p_est = new_est
            self.yaw_est = new_yaw
        else:
            a = self.smooth
            self.p_est = self.p_est + a * (new_est - self.p_est)
            # yaw lives modulo 90 and the top face is square, so the calipers
            # can report yaw or yaw+90 for the same physical orientation. Blend
            # along the shorter arc with its own gain: taking the raw reading
            # lets that label flip on every look, which commands a ~90 deg wrist
            # swing and leaves err_rot permanently above any commit gate.
            self.yaw_est = float((self.yaw_est
                                  + self.yaw_smooth * _wrap90(new_yaw
                                                              - self.yaw_est))
                                 % 90.0)
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

    def desired_pose(self, z_offset):
        # Gripper-base pose that grasps the estimated object from above. With
        # the filter running this aims ahead of the estimate by lead_s, which
        # cancels the servo's own lag on a target that is still moving.
        p = self.p_est
        if self.kf is not None and self.kf.ready and self.lead_s != 0.0:
            p = self.kf.lead(self.lead_s)
        p_des = np.array([p[0], p[1], p[2] + TIP_OFFSET + z_offset])
        R_des = rotation_with_yaw(self.yaw_est, R_DOWN)
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
        J = self.arm.kin.jacobian(self.data.qpos[:ARM_DOF])
        v = np.concatenate([self.kp * err_p, self.ko * err_o])   # m/s, rad/s
        # Cap the commanded twist. The law asks for kp*err outright, which is
        # metre-per-second scale at the start of an approach, and dq_max does
        # not catch it. Scaling the whole 6-vector preserves the motion
        # direction and slows rotation with it.
        n = float(np.linalg.norm(v[:3]))
        if n > 1e-12:
            limit = self.v_max if v_max is None else v_max
            if np.isfinite(limit) and n > limit:
                v = v * (limit / n)
        dq = J.T @ np.linalg.solve(J @ J.T + self.damp * np.eye(6), v)
        dq = dq * self.model.opt.timestep
        return np.clip(dq, -self.dq_max, self.dq_max)

    def preposition(self, z_offset, viewer=None):
        # Coarse open-loop move to the desired pose, before servoing.
        p_des, R_des = self.desired_pose(z_offset)
        q = self.arm.solve_ik(p_des, self.data.qpos[:ARM_DOF], R_target=R_des)
        self.arm.move(q, GRIP_OPEN, rate=0.20, viewer=viewer)

    # LOOP
    def servo(self, z_offset, max_iters=1500, viewer=None, estop=None,
              aborted=None, trace=None, v_max=None, check_occlusion=False,
              track=False, stationary_s=None, search=None,
              track_tol=None, track_tol_rot=None):
        # Servo to the grasp pose at z_offset. Returns (converged, iters, err).
        data, model = self.data, self.model
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
        self._still_ref_t = float(data.time)
        if track_tol is None:
            track_tol = self.track_tol
        if track_tol_rot is None:
            track_tol_rot = self.track_tol_rot

        if self.p_est is None and not self.observe():
            return False, 0, float('inf')

        q_cmd = np.array(data.qpos[:ARM_DOF], float)
        err_norm = float('inf')
        last_t = float(data.time)

        for k in range(max_iters):
            if aborted is not None and aborted():
                self._control('open')
                return False, k, err_norm

            dt = float(data.time) - last_t
            last_t = float(data.time)

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
                    last_t = float(data.time)
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
                self.kf.advance_to(float(data.time))

            p_des, R_des = self.desired_pose(z_offset)
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
                        self._still_ref_t = float(data.time)
                    elif float(np.linalg.norm(p_now - self._still_ref)) \
                            > self.still_tol:
                        self._still_ref = p_now
                        self._still_ref_t = float(data.time)
                    self.still_time = float(data.time) - self._still_ref_t

            ov = getattr(viewer, 'overlay', None)
            if ov is not None:
                ov.set_target(p_des, R_des)
            push(viewer, 'servo',
                 'z_off %.2f  iter %d/%d' % (z_offset, k, max_iters),
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
                    'z_offset': z_offset,
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
            # Command + tick through the arm's backend, the same path Arm.move
            # uses. The servo no longer needs to know what a control cycle is.
            self.arm.backend.command(q_cmd, GRIP_OPEN)

            # Advance the world by exactly one physics step, alongside the tick.
            # This must happen per physics step, NOT per render sync: the
            # servo steps physics every iteration but only syncs every
            # SYNC_STEPS, so driving the object from the sync would move it at
            # 1/SYNC_STEPS of its real speed. The world owns the motion; the
            # robot loop only gives it the tick.
            self._step_world()

            self.arm.backend.step()

            while estop is not None and estop():
                self.arm.backend.command(q_cmd, GRIP_OPEN)
                self.arm.backend.step()
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
        q_start = np.array(self.data.qpos[:ARM_DOF], float)
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

    def retreat(self, viewer=None, estop=None, aborted=None, v_cart=None):
        # Back straight up in steps until the object is whole in view again.
        # Each step is a planned move, so the retreat itself is open loop; only
        # the verdict comes from the camera. Returns (clear, steps).
        data = self.data
        if v_cart is None:
            v_cart = self.retreat_v
        p_cur, R_cur = self.current_pose()
        anchor = self.best_p if self.best_p is not None else self.p_est
        for j in range(1, self.retreat_max + 1):
            if aborted is not None and aborted():
                return False, j - 1
            target = p_cur + np.array([0.0, 0.0, self.retreat_step * j])
            q = self.arm.solve_ik(target, data.qpos[:ARM_DOF], R_target=R_cur)
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
        p_des, R_des = self.desired_pose(0.0)
        q = self.arm.solve_ik(p_des, self.data.qpos[:ARM_DOF], R_target=R_des)
        ov = getattr(self.viewer, 'overlay', None)
        if ov is not None:
            ov.set_target(p_des, R_des)
        push(viewer, 'occlusion',
             'open-loop commit to %6.3f %6.3f' % (self.best_p[0], self.best_p[1]),
             'from the fullest view, span %.4f' % self.best_span)
        # same pacing again, so the open-loop descent matches the servo it replaces
        d_cart = float(np.linalg.norm(p_des - np.array(self.data.xpos[self.base_id],
                                                       float)))

        # Watch for the target moving during the descent, on the camera only.
        self.target_moved = False
        counter = [0]

        def watch_step():
            counter[0] += 1
            if counter[0] % self.perceive_every == 0:
                self.motion_watch(viewer=viewer)

        def stop_now():
            # the descent also stops early the moment motion is seen
            return ((aborted is not None and aborted()) or self.target_moved)

        watch_step_ = watch_step if watch else None
        self._paced_move(q, d_cart, self.retreat_v, viewer=viewer, estop=estop,
                         aborted=stop_now, watch=watch_step_)
        if self.target_moved:
            return False, 0
        if aborted is not None and aborted():
            return False, 0
        return True, 1

    def approach(self, standoff=0.10, viewer=None, estop=None, aborted=None,
                 trace=None, coarse=True, max_iters=1500, recover=True,
                 check_occlusion=True, track=None, search=None,
                 stationary_s=None, track_iters=None):
        # Coarse move to the hover pose, then closed-loop hover and descent.
        if self.p_est is None and not self.observe():
            return False, 0, float('inf'), 0
        if track is None:
            track = bool(self.track_moving)
        if stationary_s is None:
            stationary_s = self.still_time_target

        if track:
            # Keep servoing for as long as the object keeps moving. The stage
            # ends when the target has been watched still long enough, at which
            # point the grasp is driven open loop from the hover -- where the
            # view is still whole -- instead of servoing down into the jaws'
            # own shadow.
            if coarse:
                self.preposition(standoff, viewer=viewer)
            n_done = 0
            for attempt in range(max(1, self.commit_tries)):
                ok1, n1, e1 = self.servo(
                    standoff, max_iters=(track_iters or self.track_iters),
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
            self.preposition(standoff, viewer=viewer)
        ok1, n1, e1 = self.servo(standoff, max_iters=max_iters, viewer=viewer,
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
        ok2, n2, e2 = self.servo(0.0, max_iters=max_iters, viewer=viewer,
                                 estop=estop, aborted=aborted, trace=trace,
                                 v_max=self.v_descent,
                                 check_occlusion=check_occlusion)
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
