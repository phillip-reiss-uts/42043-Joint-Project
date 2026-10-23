import time, numpy as np
from src.robot_backend_ur import connect
from src.arm import Arm
from src.arm_dynamics import ArmDynamics
from src.kinematics_ur import URKinematics
from src import config as cfg
from src.config import FEED, R_DOWN

b = connect(cfg.UR_IP, activate_gripper=False)
b._gripper = None
dyn = ArmDynamics.from_file('models/ur3e/arm_dynamics.json')
kin = URKinematics(dyn, (np.full(6, cfg.UR_SERVO_KP), np.full(6, cfg.UR_SERVO_KV)),
                   tool_pos=cfg.TOOL_MOUNT_POS, tool_quat=cfg.TOOL_MOUNT_QUAT,
                   root_yaw=cfg.UR_ROOT_YAW)
arm = Arm(None, None, kinematics=kin, backend=b)

q_home = np.array(b.joint_positions(), float)
p0, R0 = kin.pose(q_home)
print('start tool (mm):', np.round(p0*1000, 1))
print('start tilt (deg): %.1f' % np.degrees(np.arccos(np.clip(-R0[:,2][2], -1, 1))))
print()

def goto(p, q_seed):
    q = arm.solve_ik(p, q_seed, R_target=R0)   # hold the start orientation
    p_ach, _ = kin.pose(q[:6])
    err = np.linalg.norm(p_ach - p)*1000
    arm.move(q, 0.0, rate=FEED)
    return q, p_ach, err

D = 0.020
results = []
for axis, name in ((0, '+x'), (1, '+y'), (2, '+z')):
    for sign, sname in ((+1, '+'), (-1, '-')):
        tgt = p0.copy(); tgt[axis] += sign*D
        q, pach, ikerr = goto(tgt, b.joint_positions())
        results.append((name+sname, p0, tgt, pach, ikerr))
        print('%-3s  target %s  reached %s  ik_err %.2f mm'
              % (sname+name[1], np.round(tgt*1000,1), np.round(pach*1000,1), ikerr))
        # return to the start pose before the next axis
        goto(p0, b.joint_positions())

# repeatability: same pose five times
print()
print('repeatability (5 returns to start):')
pts = []
for i in range(5):
    goto(p0 + np.array([0, 0, 0.03]), b.joint_positions())   # move away
    arm.move(arm.solve_ik(p0, b.joint_positions(), R_target=R0), 0.0, rate=FEED)
    time.sleep(0.3)                                          # let it settle
    q_real = b.joint_positions()                             # <-- the actual arm
    p_real, _ = kin.pose(q_real)
    pts.append(p_real)
    print('  %d: %s mm' % (i+1, np.round(p_real*1000, 2)))

pts = np.array(pts)
print('spread: %s mm' % np.round(np.ptp(pts, axis=0)*1000, 2))
b.close()