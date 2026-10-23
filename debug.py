import numpy as np
from rtde_receive import RTDEReceiveInterface

r = RTDEReceiveInterface('192.168.0.194')
q  = np.array(r.getActualQ())
tcp = np.array(r.getActualTCPPose()[:3])
r.disconnect()

from src.arm_dynamics import ArmDynamics
from src.kinematics_ur import URKinematics
from src import config as cfg

dyn = ArmDynamics.from_file('models/ur3e/arm_dynamics.json')
kin = URKinematics(dyn, (np.full(6, cfg.UR_SERVO_KP), np.full(6, cfg.UR_SERVO_KV)),
                   tool_pos=cfg.TOOL_MOUNT_POS, tool_quat=cfg.TOOL_MOUNT_QUAT)

p, R = kin.pose(q)                      # gripper base, base frame
print('q      :', np.round(q, 4))
print('TCP    :', np.round(tcp*1000, 1), 'mm')      # flange
print('FK     :', np.round(p*1000, 1), 'mm')        # gripper base
print('d      :', np.round((p - tcp)*1000, 1), 'mm')
print('|d|    : %.1f mm' % (np.linalg.norm(p - tcp)*1000))