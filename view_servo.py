# view_servo.py -- animate the PBVS servo loop. Writes servo.gif (and .mp4).
#
# Two panels, following the lecture's IBVS animation: the image plane on the
# left, the camera in the world on the right. Every frame is a recorded
# iteration of a real PBVSServo.approach() run, so the convergence shown is
# the loop's own.
import io
import contextlib
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import mujoco

from src.model_loader import load_robot_model
from src.camera import RGBDCamera
from src.perception import find_camera
from src.arm import Arm
from src.pick_place import ObjectRegistry, reset_scene, place_cube
from src.servo import PBVSServo
from src.config import (
    POOL_SIZE, TABLE_HEIGHT, TABLE_CENTRE, TABLE_HALF, OBJECT_SIZE,
)

SEED = 1
STANDOFF = 0.10
MAX_FRAMES = 160
EDGE = float(np.asarray(OBJECT_SIZE)[0]) * 2.0
IMG_W, IMG_H = 640, 480


def project(p_world, cam_pos, cam_mat, K):
    # Inverse of RGBDCamera.deproject. MuJoCo cameras look down -z.
    p_cam = cam_mat.T @ (np.asarray(p_world, float) - cam_pos)
    z = -p_cam[2]
    if z <= 1e-6:
        return None
    return np.array([K[0, 2] - p_cam[0] * K[0, 0] / z,
                     K[1, 2] - p_cam[1] * K[1, 1] / z])


def frustum(cam_pos, cam_mat, depth=0.18):
    # Wireframe frustum, built in the camera frame then mapped to the world.
    half_h = depth * np.tan(np.radians(30.0))
    half_w = half_h * (float(IMG_W) / float(IMG_H))
    local = np.array([[0.0, 0.0, 0.0],
                      [-half_w, +half_h, -depth],
                      [+half_w, +half_h, -depth],
                      [+half_w, -half_h, -depth],
                      [-half_w, -half_h, -depth]])
    world = (cam_mat @ local.T).T + cam_pos
    order = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 2, 3, 4, 1]
    return world[order]


def run():
    model = load_robot_model()
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    name, cam_id = find_camera(model)
    camera = RGBDCamera(model, name, IMG_W, IMG_H)
    arm = Arm(model, data)
    reg = ObjectRegistry(model, POOL_SIZE)

    # Not every hard-mode spawn is perceivable from the home pose: a partly
    # occluded cube produces a clipped cluster that the span guard rejects, so
    # observe() yields nothing and the servo has no estimate to start from.
    # Re-seed until the first view succeeds.
    obj_true = None
    servo = None
    for attempt, seed in enumerate(range(SEED, SEED + 40)):
        rng = np.random.default_rng(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            reset_scene(model, data, reg, rng, viewer=None)
            idx = place_cube(model, data, reg, rng, settle=True)
        candidate = PBVSServo(model, data, arm, camera, cam_id)
        if candidate.observe():
            servo, obj_true = candidate, reg.pose(data, idx)
            break
        print('  seed %d: cube at %s not perceivable from home, retrying'
              % (seed, np.round(reg.pose(data, idx)[:2], 4)))
    if servo is None:
        raise SystemExit('no perceivable spawn in 40 seeds')

    # The camera's fixed offset in the gripper-base frame, measured once here.
    Rb = np.array(data.xmat[arm.base_id], float).reshape(3, 3)
    pb = np.array(data.xpos[arm.base_id], float)
    loc_R = Rb.T @ np.array(data.cam_xmat[cam_id], float).reshape(3, 3)
    loc_t = Rb.T @ (np.array(data.cam_xpos[cam_id], float) - pb)

    # coarse=False: no open-loop preposition, so the closed loop drives the
    # whole approach from home and the animation shows the actual convergence.
    trace = []
    ok, iters, err, hover = servo.approach(standoff=STANDOFF, trace=trace,
                                           coarse=False)

    print()
    print('=' * 74)
    print('PBVS SERVO ANIMATION')
    print('=' * 74)
    print('spawn              : seed offset %d' % attempt)
    print('object true centre : %s' % np.round(obj_true, 4))
    print('visual estimate    : %s  (yaw %.1f deg)'
          % (np.round(servo.p_est, 4), servo.yaw_est))
    print('estimate error     : %.4f m'
          % np.linalg.norm(servo.p_est[:2] - obj_true[:2]))
    print('converged          : %s in %d iterations (%d hover)'
          % (ok, iters, hover))
    print('final pose error   : %.5f m' % err)
    print('trace entries      : %d' % len(trace))
    camera.renderer.close()
    return camera, trace, servo, obj_true, loc_R, loc_t


def animate(camera, trace, servo, obj_true, loc_R, loc_t):
    K = camera.K
    step = max(1, len(trace) // MAX_FRAMES)
    frames = list(range(0, len(trace), step))
    if frames[-1] != len(trace) - 1:
        frames.append(len(trace) - 1)

    goal = trace[-1]
    goal_cam_pos = np.array(goal['p_des'], float) + np.array(goal['R_des'],
                                                             float) @ loc_t
    goal_cam_mat = np.array(goal['R_des'], float) @ loc_R
    tgt = project(goal['p_est'], goal_cam_pos, goal_cam_mat, K)

    def img_pt(f):
        return project(f['p_est'], np.array(f['cam_pos'], float),
                       np.array(f['cam_mat'], float), K)

    pts = []
    for i in frames:
        p = img_pt(trace[i])
        pts.append([np.nan, np.nan] if p is None else p)
    pts = np.array(pts)

    cam_track = np.array([f['cam_pos'] for f in trace])
    th = np.radians(servo.yaw_est)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    rotm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sq = (corners * EDGE / 2.0) @ rotm.T + obj_true[:2]
    x0, x1 = TABLE_CENTRE[0] - TABLE_HALF[0], TABLE_CENTRE[0] + TABLE_HALF[0]
    y0, y1 = TABLE_CENTRE[1] - TABLE_HALF[1], TABLE_CENTRE[1] + TABLE_HALF[1]
    table = np.array([[x0, y0, TABLE_HEIGHT], [x1, y0, TABLE_HEIGHT],
                      [x1, y1, TABLE_HEIGHT], [x0, y1, TABLE_HEIGHT],
                      [x0, y0, TABLE_HEIGHT]])

    fig = plt.figure(figsize=(13, 6))
    axi = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')

    def draw(j):
        i = frames[j]
        f = trace[i]
        axi.cla()
        ax3.cla()

        # image plane
        axi.plot(pts[:j + 1, 0], pts[:j + 1, 1], '-', color='C0', lw=1.8)
        axi.plot(pts[0, 0], pts[0, 1], 'o', color='C0', ms=10, mfc='none',
                 label='start')
        axi.plot(pts[j, 0], pts[j, 1], 'o', color='C0', ms=10,
                 label='object now')
        if tgt is not None:
            axi.plot(tgt[0], tgt[1], '*', color='C3', ms=18,
                     label='object at goal pose')
        z = float(np.linalg.norm(np.array(f['p_est']) - np.array(f['cam_pos'])))
        r_px = servo.tol_p * abs(K[1, 1]) / max(z, 1e-6)
        axi.add_patch(plt.Circle(pts[j], r_px, fill=False, ec='C3', ls='--',
                                 lw=1.2))
        axi.plot(K[0, 2], K[1, 2], '+', color='0.5', ms=12, mew=1.2,
                 label='principal point')
        axi.add_patch(plt.Rectangle((0, 0), IMG_W, IMG_H, fill=False,
                                    ec='0.6', lw=1))
        axi.set_xlim(-20, IMG_W + 20)
        axi.set_ylim(IMG_H + 20, -20)
        axi.set_aspect('equal')
        axi.set_xlabel('u (pixels)')
        axi.set_ylabel('v (pixels)')
        axi.set_title('image plane   step %d   pos err %.4f m   '
                      'tol %.1f px' % (f['k'], f['err_p'], r_px))
        axi.legend(fontsize=7, loc='upper right')

        # camera in the world
        ax3.plot(table[:, 0], table[:, 1], table[:, 2], color='0.7', lw=1)
        ax3.plot(sq[:, 0], sq[:, 1], np.full(len(sq), TABLE_HEIGHT),
                 color='C3', lw=2)
        ax3.plot(cam_track[:i + 1, 0], cam_track[:i + 1, 1],
                 cam_track[:i + 1, 2], 'k-', lw=1)
        fr = frustum(np.array(f['cam_pos'], float),
                     np.array(f['cam_mat'], float))
        ax3.plot(fr[:, 0], fr[:, 1], fr[:, 2], color='C0', lw=1.3)
        ax3.plot(*goal['p_des'], marker='*', color='C3', ms=14,
                 label='grasp goal')
        ax3.set_xlim(-0.52, 0.10)
        ax3.set_ylim(0.28, 0.88)
        ax3.set_zlim(0.0, 0.62)
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('y (m)')
        ax3.set_zlabel('z (m)')
        ax3.set_box_aspect((1.0, 0.8, 0.7))
        ax3.view_init(22, -60)
        ax3.set_title('camera in the world   z_offset %.2f' % f['z_offset'])
        ax3.legend(fontsize=7, loc='upper left')

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
    out = 'servo.gif'
    anim.save(out, writer=PillowWriter(fps=12))
    print('wrote %s  (%d frames, %.1f KB)'
          % (out, len(frames), os.path.getsize(out) / 1024.0))
    try:
        anim.save('servo.mp4', writer='ffmpeg', fps=12)
        print('wrote servo.mp4  (%.1f KB)'
              % (os.path.getsize('servo.mp4') / 1024.0))
    except Exception as exc:
        print('mp4 skipped (%s)' % type(exc).__name__)
    plt.close(fig)


if __name__ == '__main__':
    camera, trace, servo, obj_true, loc_R, loc_t = run()
    animate(camera, trace, servo, obj_true, loc_R, loc_t)
