from src.robot_backend_ur import connect
from src import config as cfg
import numpy as np, time

b = connect(cfg.UR_IP, activate_gripper=False)
b._gripper = None
q0 = b.joint_positions()
tgt = q0.copy(); tgt[0] += np.radians(2)
q = [float(v) for v in tgt]

# try the positional form by hand and print what it returns
fn = b._control.servoJ
t0 = time.monotonic()
r = fn(q, 0.0, 0.0, 0.008, 0.1, 300.0)
print('servoJ returned:', r, 'after %.1f ms' % ((time.monotonic()-t0)*1000))
time.sleep(0.2)
print('q now:', np.round(np.degrees(b.joint_positions()), 2))
b.close()