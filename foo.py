import time
from onRobot.gripper import RG
r = RG('192.168.0.194', 0)
print('start', r.get_rg_width())
t = time.monotonic(); r.rg_grip(56.0, 3.0)
print('ELAPSED %.3f s' % (time.monotonic() - t))
print('end', r.get_rg_width())
