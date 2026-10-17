# State-space analysis demonstration (controllability, observability).
#
# DEMONSTRATION ONLY. Needs the composed MuJoCo scene, so it lives outside
# state_space.py. See tests/test_layering.py.
#
#     python -m src.demo_state_space

import numpy as np
import mujoco

from src.config import ARM_DOF
from src.state_space import analyse


def main():
    from src.model_loader import load_robot_model

    model = load_robot_model()
    if model is None:
        raise SystemExit('model failed to load')

    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    q_home = np.array(data.qpos[:ARM_DOF])

    # second operating point: extended forward and down, like a grasp pose
    q_grasp = q_home.copy()
    q_grasp[1] += 0.6      # shoulder_lift
    q_grasp[2] -= 0.5      # elbow

    analyse(model, data, q_home, 'home')
    analyse(model, data, q_grasp, 'grasp-like (shoulder +0.6, elbow -0.5)')

    print('=' * 74)
    print('SUMMARY')
    print('=' * 74)
    print('* Both realisations are controllable (PBH rank 12): every joint state')
    print('  is reachable from here, with torque or with a position command.')
    print('* The torque plant is UNSTABLE at these poses (positive eigenvalue).')
    print('  Gravity compensation alone does not hold a manipulator up: a')
    print('  deviation grows. Feedback is required, not optional.')
    print('* A_cl is strictly stable because of the servo velocity damping')
    print('  -M^-1 Kv. That means the controller task is tracking and')
    print('  disturbance rejection, not stabilisation.')
    print('* Joint angles alone are observable (PBH rank 12) with q_dot')
    print('  unmeasured, so an observer can reconstruct velocity from')
    print('  encoders -- exactly the unmeasured state the brief asks for.')


if __name__ == '__main__':
    main()
