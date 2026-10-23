import time, numpy as np
from src.robot_backend_ur import connect
from src import config as cfg

b = connect(cfg.UR_IP, activate_gripper=False)
b._gripper = None          # OnRobot is driven by the UR programs, not Modbus

q0 = b.joint_positions()
print('start :', np.round(np.degrees(q0), 2))

tgt = q0.copy()
tgt[0] += np.radians(2.0)          # shoulder_pan only

b.command(tgt, 0.0)                # <-- this must be here
t0 = time.monotonic()
for i in range(125):               # ~1 s at 125 Hz
    b.step()
    if i % 25 == 0:
        print('  t=%.2fs  q1=%6.2f deg' % (time.monotonic() - t0,
                                           np.degrees(b.joint_positions()[0])))
dt = (time.monotonic() - t0) / 125
print('end   :', np.round(np.degrees(b.joint_positions()), 2))
print('mean step: %.1f ms (expect ~8)' % (dt * 1000))
b.close()