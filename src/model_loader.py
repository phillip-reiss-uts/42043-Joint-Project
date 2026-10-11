import mujoco
import mujoco_menagerie as mm

from src.config import (
    CAM_QUAT, CAM_POS, MESH_QUAT, MESH_POS,
    TABLE_HEIGHT, TABLE_THICK, TABLE_CENTRE, TABLE_HALF,
    OBJECT_SIZE,
    # LEFTOVER: imported but never used here -- OBJECT_POS was for the old single
    # 'target' body; INITIAL_OBJECTS and MIN_OBJECT_SEPARATION are used by the
    # task layer (pick_place.py), not by model construction.
    # OBJECT_POS,
    # INITIAL_OBJECTS, MIN_OBJECT_SEPARATION,
    POOL_SIZE,
    CHUTE_X, CHUTE_Y, CHUTE_FLOOR_Z, CHUTE_WALL_THICK, CHUTE_WALL_HEIGHT,
    PARK_POS,
)


def load_robot_model():
    """Compose UR5e + Robotiq 2F-85 + Realsense D435i + table + chute + object pool.

    Returns a compiled MjModel, or None on failure.
    """
    try:
        arm = mm.get('universal_robots_ur5e').spec('ur5e')
        gripper = mm.get('robotiq_2f85').spec('2f85')
        camera = mm.get('realsense_d435i').spec('d435i')
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print("Ensure 'mujoco-menagerie' is installed: pip install mujoco-menagerie")
        return None

    # ------------------------------------------------------------------
    # 1. Attach gripper to the UR5e flange
    # ------------------------------------------------------------------
    attachment_site = arm.site("attachment_site")
    if attachment_site is None:
        print("[ERROR] Could not find 'attachment_site' in UR5e model.")
        return None
    arm.attach(gripper, prefix="gripper/", site=attachment_site)

    # ------------------------------------------------------------------
    # 2. Attach D435i mesh to the gripper base via a frame
    # ------------------------------------------------------------------
    gripper_base = arm.body("gripper/base")
    if gripper_base is None:
        gripper_base = arm.body("base")

    gripper_frame = gripper_base.add_frame()
    gripper_frame.pos = MESH_POS
    gripper_frame.quat = MESH_QUAT

    arm.attach(camera, prefix="camera/", frame=gripper_frame)

    # ------------------------------------------------------------------
    # 3. Camera element on the gripper base (must come after attach)
    # ------------------------------------------------------------------
    gripper_base.add_camera(
        name="depth_cam",
        pos=CAM_POS,
        quat=CAM_QUAT,
        fovy=60.0,
    )
    print(f"[INFO] Added 'depth_cam' to {gripper_base.name}")

    # ------------------------------------------------------------------
    # 4. Ground plane
    # ------------------------------------------------------------------
    floor_geom = arm.worldbody.add_geom()
    floor_geom.name = "floor"
    floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor_geom.size = [10, 10, 0.1]
    floor_geom.pos = [0, 0, -0.50]
    #floor_geom.pos = [0, 0, 0]

    plinth = arm.worldbody.add_geom()
    plinth.name = "table_plinth"
    plinth.type = mujoco.mjtGeom.mjGEOM_BOX
    plinth.pos = [TABLE_CENTRE[0], TABLE_CENTRE[1], -0.25]
    plinth.size = [TABLE_HALF[0], TABLE_HALF[1], 0.25]
    plinth.rgba = [0.45, 0.38, 0.30, 1.0]

    # ------------------------------------------------------------------
    # 5. Work surface. Top face lands exactly on TABLE_HEIGHT.
    # ------------------------------------------------------------------
    table_geom = arm.worldbody.add_geom()
    table_geom.name = "table"
    table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    table_geom.size = [TABLE_HALF[0], TABLE_HALF[1], TABLE_THICK / 2.0]
    table_geom.pos = [TABLE_CENTRE[0], TABLE_CENTRE[1], TABLE_HEIGHT - TABLE_THICK / 2.0]
    table_geom.rgba = [0.55, 0.45, 0.35, 1.0]

    # ------------------------------------------------------------------
    # 6. Chute: a sunk bin attached to the right edge of the table.
    #    Floor at CHUTE_FLOOR_Z, walls rising to the tabletop. Objects
    #    dropped over it fall in and are despawned by the task code.
    # ------------------------------------------------------------------
    chx0, chx1 = CHUTE_X
    chy0, chy1 = CHUTE_Y
    chute_cx = 0.5 * (chx0 + chx1)
    chute_cy = 0.5 * (chy0 + chy1)

    chute_floor = arm.worldbody.add_geom()
    chute_floor.name = "chute_floor"
    chute_floor.type = mujoco.mjtGeom.mjGEOM_BOX
    chute_floor.size = [0.5 * (chx1 - chx0), 0.5 * (chy1 - chy0), 0.01]
    chute_floor.pos = [chute_cx, chute_cy, CHUTE_FLOOR_Z - 0.01]
    chute_floor.rgba = [0.25, 0.30, 0.40, 1.0]

    wall_h = CHUTE_WALL_HEIGHT
    wall_t = CHUTE_WALL_THICK
    wall_z = CHUTE_FLOOR_Z + wall_h / 2.0

    def add_wall(name, x, y, sx, sy):
        g = arm.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [sx, sy, wall_h / 2.0]
        g.pos = [x, y, wall_z]
        g.rgba = [0.20, 0.45, 0.50, 1.0]

    # inner wall (table side): flush against the table's right edge
    add_wall("chute_wall_inner", 0.216 + wall_t, chute_cy, wall_t, 0.5 * (chy1 - chy0))
    # outer wall
    add_wall("chute_wall_outer", 0.566 - wall_t, chute_cy, wall_t, 0.5 * (chy1 - chy0))
    # end walls
    add_wall("chute_wall_near", chute_cx, chy0 - wall_t, 0.5 * (chx1 - chx0), wall_t)
    add_wall("chute_wall_far", chute_cx, chy1 + wall_t, 0.5 * (chx1 - chx0), wall_t)

    # ------------------------------------------------------------------
    # 7. Object pool: POOL_SIZE bodies, each with its own freejoint.
    #    All start parked below the scene; the task code spawns them.
    # ------------------------------------------------------------------
    for i in range(POOL_SIZE):
        b = arm.worldbody.add_body()
        b.name = f"obj_{i}"
        b.pos = PARK_POS

        j = b.add_freejoint()
        j.name = f"obj_{i}_free"

        g = b.add_geom()
        g.name = f"obj_{i}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = OBJECT_SIZE
        g.rgba = [0.9, 0.3, 0.2, 1.0]
        g.contype = 1
        g.conaffinity = 1

    # ------------------------------------------------------------------
    # 8. Gravity compensation: everything except the pooled objects.
    # ------------------------------------------------------------------
    for body in arm.bodies:
        if not body.name.startswith("obj_"):
            body.gravcomp = 1.0

    # ------------------------------------------------------------------
    # 9. Compile
    # ------------------------------------------------------------------
    try:
        model = arm.compile()
    except Exception as e:
        print(f"[ERROR] Failed to compile composed model: {e}")
        return None

    return model