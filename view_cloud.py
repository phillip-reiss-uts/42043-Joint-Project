# view_cloud.py — visualise the wrist-camera point cloud. Writes pointcloud.png.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from src.model_loader import load_robot_model
from src.camera import RGBDCamera

model = load_robot_model()
data = mujoco.MjData(model)
home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home)
mujoco.mj_forward(model, data)                # REQUIRED: populates cam_xmat/cam_xpos

cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
t = np.array(data.cam_xpos[cid]); R = np.array(data.cam_xmat[cid]).reshape(3, 3)

cam = RGBDCamera(model, "depth_cam", 640, 480)
rgb, depth = cam.capture(data)
H, W = depth.shape
#uu, vv = np.meshgrid(np.arange(W, float), np.arange(H, float))
uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

z = depth.astype(np.float64)
valid = np.isfinite(depth) & (depth > 0)
fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]

def cloud(zsign):
    pc = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, zsign * z], -1)
    pc[~valid] = np.nan
    return (pc.reshape(-1, 3) @ R.T + t).reshape(H, W, 3)

cur, fix = cloud(+1.0), cloud(-1.0)

fig = plt.figure(figsize=(15, 9))

ax = fig.add_subplot(2, 3, 1); ax.imshow(rgb); ax.set_title("RGB"); ax.axis("off")
ax = fig.add_subplot(2, 3, 2); im = ax.imshow(depth, cmap="turbo")
ax.set_title("depth (m)"); plt.colorbar(im, ax=ax, fraction=0.046)

for col, (P, name) in enumerate([(cur, "CURRENT z_cam=+depth"), (fix, "FIXED z_cam=-depth")]):
    Q = P[np.isfinite(P).all(axis=-1)]
    c = Q[:, 2]
    a = fig.add_subplot(2, 3, 3 + col)                       # side view
    a.scatter(Q[:, 0], Q[:, 2], c=c, s=0.2, cmap="viridis")
    a.axhline(0, color="k", ls="--", lw=1.2, label="floor z=0")
    a.axhline(t[2], color="r", ls=":", lw=1.2, label=f"camera z={t[2]:.3f}")
    a.set_ylim(-0.3, 1.3); a.set_xlabel("world x (m)"); a.set_ylabel("world z (m)")
    a.set_title(f"{name}\nSIDE VIEW  (x vs z)"); a.legend(fontsize=7)

    b = fig.add_subplot(2, 3, 4 + col)                       # top-down view
    b.scatter(Q[:, 0], Q[:, 1], c=c, s=0.2, cmap="viridis")
    b.plot(t[0], t[1], "r*", ms=14, label="camera")
    b.set_aspect("equal"); b.set_xlabel("world x (m)"); b.set_ylabel("world y (m)")
    b.set_title(f"{name}\nTOP-DOWN (x vs y)"); b.legend(fontsize=7)

    print(f"{name}: z [{c.min():.3f}, {c.max():.3f}]  on floor {(np.abs(c) < 5e-3).mean():.1%}")

fig.suptitle("Wrist RGB-D point cloud")
fig.tight_layout()
fig.savefig("pointcloud.png", dpi=110)
print("wrote pointcloud.png")
