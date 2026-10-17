"""Rigid-body dynamics for a 6R arm, independent of any simulator.

WHY
---
`src/kinematics.py` needs `M(q)` and `g(q)` on hardware. MuJoCo gives them for
free; a UR3e controller does not. This module computes them from published link
data with a standard recursive Newton-Euler formulation, so the observer and the
linearisation do not depend on the simulator.

FORMULATION (composite rigid body)
----------------------------------
With the standard convention (link frames at joint origins, joint axis z_i in
the link frame i):

  forward pass   R_i, p_i, z_i for every link from q.

  M(q)           column j is the torque felt about each joint i <= j when link j
                 (and everything distal to it) is accelerated at unit rate about
                 joint j. Concretely, for the distal set D(j):

                     J_j = sum_{k in D(j)} I_k^com + m_k S(r_k) S(r_k)^T
                     M[i,j] = z_i . (J_j_shifted) ...

                 Rather than shift composite inertias between frames, this
                 implementation evaluates the equivalent quantity directly:

                     M[i,j] = sum_{k in D(j)} z_i . [ I_k z_j + r_ki x (m_k (z_j x r_kj)) ]

                 where r_kj = com_k - p_j and r_ki = com_k - p_i. This is the
                 wrench of the distal set under unit acceleration about z_j,
                 projected onto z_i. O(n^3) worst case; n=6 makes this free and
                 it is far easier to verify than the O(n) recursive form.

  g(q)           the torque needed to hold q. Each link's weight acts at its
                 com; joint i supplies the moment of every distal link about
                 its axis.

ROTOR INERTIA IS NOT OPTIONAL
-----------------------------
A real UR joint reflects its motor and gearbox inertia through the reducer,
which appears as a diagonal term added to M. It is NOT in the URDF, and it is
large: on the UR5e model used here `dof_armature = 0.1 kg m^2` per joint, while
the wrist links have inertia of only ~0.0025. Ignoring it makes the wrist
diagonal wrong by ~40x.

It is a required constructor argument rather than a defaulted one, precisely so
a caller has to think about it. Identify it on the real arm.
"""

import numpy as np


def _skew(w):
    w = np.asarray(w, float)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def rot_from_quat(q):
    # Quaternion (w, x, y, z) -> rotation matrix.
    w, x, y, z = np.asarray(q, float)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def rot_axis(axis, angle):
    # Rodrigues rotation about a unit axis.
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(angle), np.sin(angle)
    K = _skew(a)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


class Link:
    # One moving link: geometry relative to its parent, plus inertial data.
    #
    # `inertia` may be a (3,) vector of principal moments (assumed already
    # expressed in the link frame) or a full (3,3) tensor. UR URDFs give
    # principal moments plus an `iquat`; `Link.from_model` folds that rotation in
    # so the tensor below is always in the link frame.

    def __init__(self, name, pos, quat, mass, com, inertia, axis,
                 inert_pos=None):
        self.name = name
        self.pos = np.asarray(pos, float)          # origin in parent frame
        self.R = rot_from_quat(quat)               # orientation in parent frame
        self.mass = float(mass)
        self.com = np.asarray(com, float)          # centre of mass, link frame
        I = np.asarray(inertia, float)
        self.I_tensor = np.diag(I) if I.shape == (3,) else I
        self.axis = np.asarray(axis, float)        # joint axis, link frame

    @classmethod
    def from_model(cls, model, body_id, joint_axis, mu=None):
        # Build a Link from a MuJoCo body/joint pair. Convenience for the
        # simulation provider; a hardware build constructs Links from URDF data
        # directly.
        import mujoco
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        principal = np.asarray(model.body_inertia[body_id], float)
        iquat = np.asarray(model.body_iquat[body_id], float)
        # Rotate principal moments into the link frame.
        R = rot_from_quat(iquat)
        I = R @ np.diag(principal) @ R.T
        return cls(name=name,
                   pos=model.body_pos[body_id],
                   quat=model.body_quat[body_id],
                   mass=model.body_mass[body_id],
                   com=model.body_ipos[body_id],
                   inertia=I,
                   axis=joint_axis)


_JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


def _root_transform(model, first_link, n):
    # Rotation from world to the first link's parent body frame.
    import mujoco
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _JOINT_NAMES[0])
    parent = int(model.body_parentid[int(model.jnt_bodyid[jnt])])
    q = np.asarray(model.body_quat[parent], float)
    return rot_from_quat(q)


def _merge_fixed_children(model, body, parent_link):
    # Combine a body's rigidly-attached subtree into one mass distribution,
    # expressed in the parent link's frame.
    #
    # Done in world coordinates at a reference configuration, then converted
    # back once: composing relative transforms child-by-child is easy to get
    # subtly wrong, and a wrong com silently corrupts g(q).
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    # The parent link's own body, to define the output frame.
    members = []
    frontier = [body]
    while frontier:
        b = frontier.pop()
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ''
        if nm.startswith('obj_'):
            continue                      # pooled objects are not the robot
        members.append(b)
        for child in range(1, model.nbody):
            if model.body_parentid[child] == b:
                frontier.append(child)

    # First pass: total mass and the merged centre of mass.
    total = 0.0
    com_world = np.zeros(3)
    for b in members:
        m_b = float(model.body_mass[b])
        if m_b <= 0.0:
            continue
        total += m_b
        com_world += m_b * np.asarray(data.xipos[b], float)
    if total <= 0.0:
        return parent_link.mass, parent_link.com, parent_link.I_tensor
    com_world /= total

    # Second pass: inertia about the merged com. Each body contributes its own
    # inertia plus the parallel-axis term for the offset from the merged com --
    # NOT from the world origin, which would be double-counting.
    I_world = np.zeros((3, 3))
    for b in members:
        m_b = float(model.body_mass[b])
        if m_b <= 0.0:
            continue
        R = np.asarray(data.xmat[b], float).reshape(3, 3)
        Iq = rot_from_quat(model.body_iquat[b])
        I_b = R @ Iq @ np.diag(model.body_inertia[b]) @ Iq.T @ R.T
        d = np.asarray(data.xipos[b], float) - com_world
        I_world += I_b + m_b * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    # Convert back into the parent link's frame.
    Rb = np.asarray(data.xmat[body], float).reshape(3, 3)
    pb = np.asarray(data.xpos[body], float)
    com_local = Rb.T @ (com_world - pb)
    I_local = Rb.T @ I_world @ Rb
    return total, com_local, I_local


class ArmDynamics:
    # Composite rigid-body dynamics for a serial chain of revolute joints.

    @classmethod
    def from_mujoco(cls, model, n=6, tip_body=None):
        # Build a 6R chain from a loaded MuJoCo model.
        #
        # Handles three things the URDF does not make obvious:
        #   * the arm is mounted on a `base` body whose quat is a 180 deg
        #     rotation about z. Every downstream frame inherits it, so it must
        #     be folded into the first link or all the axes come out wrong;
        #   * bodies distal to the last joint with NO joint of their own (the
        #     gripper) are rigidly attached, so their mass and inertia are
        #     merged into the last moving link;
        #   * `body_iquat` may rotate a link's principal moments away from its
        #     frame, which has to be folded in before use.
        import mujoco
        links = []
        for i in range(n):
            jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    '%s' % _JOINT_NAMES[i])
            body = int(model.jnt_bodyid[jnt])
            links.append(Link.from_model(model, body, model.jnt_axis[jnt]))

        # Apply any fixed transform between the world and the first joint's
        # parent body (the UR `base`). M(q) and g(q) are in joint space and so
        # are invariant to a rigid re-orientation of the whole arm -- but the
        # gravity *direction* is expressed in world coordinates, so the chain
        # has to be expressed in the same frame MuJoCo uses.
        links[0].R = _root_transform(model, links[0], n) @ links[0].R

        # Merge every body that hangs off the last link without a joint.
        last_body = int(model.jnt_bodyid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              _JOINT_NAMES[n - 1])])
        merged_mass, merged_com, merged_I = _merge_fixed_children(
            model, last_body, links[-1])
        links[-1].mass = merged_mass
        links[-1].com = merged_com
        links[-1].I_tensor = merged_I

        rotor = np.array(model.dof_armature[:n], float)
        return cls(links, rotor_inertia=rotor, gravity=model.opt.gravity)

    def __init__(self, links, rotor_inertia, gravity=(0.0, 0.0, -9.81)):
        self.links = list(links)
        self.n = len(self.links)
        self.rotor = np.asarray(rotor_inertia, float).copy()
        if self.rotor.shape != (self.n,):
            raise ValueError('rotor_inertia needs one entry per link')
        self.gravity = np.asarray(gravity, float)

    # FORWARD KINEMATICS OF THE CHAIN
    def frames(self, q):
        # Per-link world (R, p, z, com, I_world).
        q = np.asarray(q, float)
        if q.shape != (self.n,):
            raise ValueError('q must have %d entries' % self.n)

        R_par = np.eye(3)
        p_par = np.zeros(3)
        out = []
        for i, link in enumerate(self.links):
            R = R_par @ link.R @ rot_axis(link.axis, q[i])
            p = p_par + R_par @ link.pos
            z = R @ link.axis
            com = p + R @ link.com
            I_world = R @ link.I_tensor @ R.T
            out.append(dict(R=R, p=p, z=z, com=com, I=I_world,
                            mass=link.mass))
            R_par, p_par = R, p
        return out

    # MASS MATRIX
    def mass_matrix(self, q):
        # Composite rigid body, direct form:
        #
        #   M[i,j] = sum over links k >= max(i,j) of
        #            z_i . ( I_k z_j + r_ki x (m_k (z_j x r_kj)) )
        #
        # where r_kj = com_k - p_j. Only links distal to BOTH joints
        # contribute: a link upstream of either joint is unmoved by that joint.
        #
        # An earlier version projected a wrench accumulated at joint j onto
        # joint i by shifting its moment arm. That shift double-counts the
        # lever arm and gets the sign of M[0,1] wrong, while leaving every
        # diagonal correct -- which is exactly what made it hard to spot.
        f = self.frames(q)
        n = self.n
        M = np.zeros((n, n))

        for i in range(n):
            zi, pi = f[i]['z'], f[i]['p']
            for j in range(i, n):
                zj, pj = f[j]['z'], f[j]['p']
                total = 0.0
                for k in range(j, n):          # k >= max(i, j) == j
                    bk = f[k]
                    rkj = bk['com'] - pj
                    rki = bk['com'] - pi
                    a = np.cross(zj, rkj)       # com accel per unit qdd_j
                    total += zi @ (bk['I'] @ zj
                                   + np.cross(rki, bk['mass'] * a))
                M[i, j] = total
                M[j, i] = total

        M += np.diag(self.rotor)
        return M

    # GRAVITY TORQUE
    def gravity_torque(self, q):
        # Torque to hold q: for each joint i, the moment of every distal link's
        # weight about z_i.
        f = self.frames(q)
        g = self.gravity
        tau = np.zeros(self.n)
        for i in range(self.n):
            zi, pi = f[i]['z'], f[i]['p']
            total = 0.0
            for k in range(i, self.n):
                bk = f[k]
                r = bk['com'] - pi
                total += zi @ np.cross(r, bk['mass'] * g)
            # The joint must supply the negative of the gravity moment.
            tau[i] = -total
        return tau

    def mass_gravity(self, q):
        # One kinematic pass for both; see mass_matrix for the formula.
        f = self.frames(q)
        n = self.n

        M = np.zeros((n, n))
        for i in range(n):
            zi, pi = f[i]['z'], f[i]['p']
            for j in range(i, n):
                zj, pj = f[j]['z'], f[j]['p']
                total = 0.0
                for k in range(j, n):
                    bk = f[k]
                    rkj = bk['com'] - pj
                    rki = bk['com'] - pi
                    a = np.cross(zj, rkj)
                    total += zi @ (bk['I'] @ zj
                                   + np.cross(rki, bk['mass'] * a))
                M[i, j] = total
                M[j, i] = total
        M += np.diag(self.rotor)

        g = self.gravity
        tau = np.zeros(n)
        for i in range(n):
            zi, pi = f[i]['z'], f[i]['p']
            total = 0.0
            for k in range(i, n):
                bk = f[k]
                total += zi @ np.cross(bk['com'] - pi, bk['mass'] * g)
            tau[i] = -total

        return M, tau
