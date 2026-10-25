"""Rebuild models/ur3e/arm_dynamics.json from the composed MuJoCo model.

WHY THIS EXISTS
    URKinematics (the hardware chain) cannot call MuJoCo, so the arm's link
    dynamics are shipped as data: mass, centre of mass, inertia, joint axis per
    link. That file must describe the SAME plant the simulator composes --
    otherwise the chain's M(q) and g(q) quietly disagree with the model, which is
    what tests/test_ur_kinematics.py measures.

    It is NOT hand-written and must never be hand-edited. Changing the payload
    (gripper, camera, mount) invalidates it.

WHAT GETS MERGED, AND WHY
    Links 0-4 map one-to-one onto MJCF bodies. The last link is the exception:
    everything rigidly bolted downstream of wrist_3 becomes part of link 5, so the
    chain sees one rigid body where the model has several.

    The merge set that reproduces the model exactly is:

        wrist_3_link + gripper/base + both finger bodies + camera/d435i

    Leaving the camera out is the subtle one: it is only ~72 g, but it sits about
    100 mm off the wrist, and dropping it puts |dg| at 2.6e-1 N.m -- five orders
    too big. Verified by sweeping the four plausible merge sets and keeping the one
    that lands at float noise (|dM| 1e-15, |dg| 9e-15).

    The fingers are merged RIGIDLY at their qpos=0 (fully open) pose. The real
    jaw opening changes the payload's inertia slightly; that is a deliberate
    approximation, and it is why this file is regenerated rather than tuned.

USAGE
    cd "assignment 4"
    python tools/build_arm_dynamics.py           # rewrite the JSON
    python tests/test_ur_kinematics.py           # must pass
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from src.model_loader import load_robot_model  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(PROJECT, 'models', 'ur3e', 'arm_dynamics.json')

LAST_LINK = 'wrist_3_link'
MERGE = [
    # The RG2's real links: palm plus four knuckles and two inner fingers. They
    # are merged RIGIDLY at the open pose, which is an approximation -- the real
    # linkage moves -- but the wrist load is what the chain needs.
    'gripper/base',
    'gripper/left_outer_knuckle',
    'gripper/left_inner_knuckle',
    'gripper/left_inner_finger',
    'gripper/right_outer_knuckle',
    'gripper/right_inner_knuckle',
    'gripper/right_inner_finger',
    'camera/d435i',
]


def _part(model, data, name):
    # mass, world com, world inertia tensor about that com
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise SystemExit('body %r not in the composed model' % name)
    R = np.array(data.xmat[i]).reshape(3, 3)
    p = np.array(data.xpos[i])
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, model.body_iquat[i])
    Rq = Rq.reshape(3, 3)
    I = R @ Rq @ np.diag(np.array(model.body_inertia[i])) @ Rq.T @ R.T
    return float(model.body_mass[i]), R @ np.array(model.body_ipos[i]) + p, I


def main():
    model = load_robot_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)          # jaws at qpos 0 = fully open

    parts = [_part(model, data, n) for n in [LAST_LINK] + MERGE]
    total = np.array([p[0] for p in parts]).sum()
    com_w = sum(p[0] * p[1] for p in parts) / total

    # Each body's inertia about its OWN com, shifted onto the merged com. No
    # further parallel-axis term afterwards: the shift is already to the merged
    # com, and subtracting a second one is what produced 1.7e-2 on the first try.
    I_w = np.zeros((3, 3))
    for mass, com, I in parts:
        d = com - com_w
        I_w += I + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LAST_LINK)
    R_w = np.array(data.xmat[i]).reshape(3, 3)
    p_w = np.array(data.xpos[i])

    com_l = R_w.T @ (com_w - p_w)
    I_l = R_w.T @ I_w @ R_w

    with open(PATH) as fh:
        data_json = json.load(fh)
    data_json['links'][-1].update(
        mass=float(total),
        com=[float(v) for v in com_l],
        inertia=[[float(v) for v in row] for row in I_l],
    )
    with open(PATH, 'w') as fh:
        json.dump(data_json, fh, indent=2)

    print('merged   : %s + %s' % (LAST_LINK, ' + '.join(MERGE)))
    print('mass     : %.4f kg' % total)
    print('com      : %s (last-link frame)' % np.round(com_l, 5))
    print('inertia  : diag %s' % np.round(np.diag(I_l), 6))
    print('wrote    : %s' % PATH)
    print('now run  : python tests/test_ur_kinematics.py  (must pass)')


if __name__ == '__main__':
    main()
