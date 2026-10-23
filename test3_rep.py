"""Test 3 (part 2): repeatability — return to one pose five times, reading the ARM."""
import time
import numpy as np

from src.robot_backend_ur import connect
from src.arm import Arm
from src.arm_dynamics import ArmDynamics
from src.kinematics_ur import URKinematics
from src import config as cfg
from src.config import FEED

b = connect(cfg.UR_IP, activate_gripper=False)
b._gripper = None
dyn = ArmDynamics.from_file('models/ur3e/arm_dynamics.json')
kin = URKinematics(dyn, (np.full(6, cfg.UR_SERVO_KP), np.full(6, cfg.UR_SERVO_KV)),
                   tool_pos=cfg.TOOL_MOUNT_POS, tool_quat=cfg.TOOL_MOUNT_QUAT,
                   root_yaw=cfg.UR_ROOT_YAW)
arm = Arm(None, None, kinematics=kin, backend=b)

q_start = np.array(b.joint_positions(), float)
p0, R0 = kin.pose(q_start)
print('start tool (mm):', np.round(p0 * 1000, 1))
print()

def go_to(p):
    q = arm.solve_ik(p, b.joint_positions(), R_target=R0)
    arm.move(q, 0.0, rate=FEED)

pts = []
for i in range(5):
    go_to(p0 + np.array([0.0, 0.0, 0.030]))   # move away
    go_to(p0)                                  # come back
    time.sleep(0.3)                            # let it settle
    q_real = np.array(b.joint_positions(), float)   # <-- the actual arm
    p_real, _ = kin.pose(q_real)
    pts.append(p_real)
    d = (p_real - p0) * 1000
    print('  %d: %s mm   (offset %s)' % (i + 1, np.round(p_real * 1000, 2),
                                         np.round(d, 2)))

pts = np.array(pts)
spread = np.ptp(pts, axis=0) * 1000
print()
print('spread x/y/z: %s mm' % np.round(spread, 2))
print('max spread  : %.2f mm   (checklist: < 2 mm)' % spread.max())
print('PASS' if spread.max() < 2.0 else 'FAIL')
b.close()