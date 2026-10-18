import os

import mujoco
import mujoco_menagerie as mm

# Local model directory: <project>/models/<name>/<name>.xml
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

from src.config import (
    CAM_QUAT, CAM_POS, MESH_QUAT, MESH_POS,
    TABLE_HEIGHT, TABLE_THICK, TABLE_CENTRE, TABLE_HALF,
    OBJECT_SIZE,
    GRIP_FORCE_RANGE,
    CALIB_MARKERS, CALIB_SPACING, CALIB_MARKER_R, CALIB_BOARD_THICK,
    CALIB_BOARD_MARGIN,
    ROBOT_ARM, ROBOT_GRIPPER, ROBOT_CAMERA,
    POOL_SIZE,
    PARK_POS,
)


def _load_arm_spec(arm_model):
    # Resolve the arm spec from either a local MJCF path or mujoco_menagerie.
    #
    # The real robot is a UR3e. menagerie does not ship one, so the official
    # UR3e MJCF is vendored at models/ur3e/ur3e.xml and selected by a name that
    # is not a menagerie entry (see ROBOT_ARM in config.py). Anything else is
    # looked up in menagerie as before.
    local = os.path.join(_MODELS_DIR, arm_model, arm_model + '.xml')
    if os.path.isfile(local):
        print(f"[INFO] arm model: local MJCF {local}")
        return mujoco.MjSpec.from_file(local)
    try:
        return mm.get(arm_model).spec(arm_model.split('_')[-1])
    except Exception as e:
        raise RuntimeError(
            f"arm model '{arm_model}' not found locally at {local} and not "
            f"available in mujoco_menagerie: {e}")


def load_robot_model(with_calibration_board=False, arm_model=None,
                     gripper_model=None, camera_model=None):
    # Compose <arm> + <gripper> + <wrist camera> + table + chute + object pool.
    #
    # The arm defaults to ROBOT_ARM. The UR5e simulation this was developed
    # against is still available:
    #     load_robot_model(arm_model='universal_robots_ur5e')
    # Nothing downstream depends on which arm is chosen: joint count comes from
    # the compiled model, and no module hardcodes UR5e geometry.
    arm_model = arm_model or ROBOT_ARM
    gripper_model = gripper_model or ROBOT_GRIPPER
    camera_model = camera_model or ROBOT_CAMERA
    try:
        arm = _load_arm_spec(arm_model)
        gripper = mm.get(gripper_model).spec(gripper_model.split('_')[-1])
        camera = mm.get(camera_model).spec(camera_model.split('_')[-1])
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print("Ensure 'mujoco-menagerie' is installed: pip install mujoco-menagerie")
        return None

    # 1. Attach gripper to the arm flange
    attachment_site = arm.site("attachment_site")
    if attachment_site is None:
        print(f"[ERROR] Could not find 'attachment_site' in {arm_model} model.")
        return None
    arm.attach(gripper, prefix="gripper/", site=attachment_site)

    # 2. Attach D435i mesh to the gripper base via a frame
    gripper_base = arm.body("gripper/base")
    if gripper_base is None:
        gripper_base = arm.body("base")

    gripper_frame = gripper_base.add_frame()
    gripper_frame.pos = MESH_POS
    gripper_frame.quat = MESH_QUAT

    arm.attach(camera, prefix="camera/", frame=gripper_frame)

    # 3. Camera element on the gripper base (must come after attach)
    gripper_base.add_camera(
        name="depth_cam",
        pos=CAM_POS,
        quat=CAM_QUAT,
        fovy=60.0,
    )
    print(f"[INFO] Added 'depth_cam' to {gripper_base.name}")

    # 4. Ground plane
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

    # 5. Work surface. Top face lands exactly on TABLE_HEIGHT.
    table_geom = arm.worldbody.add_geom()
    table_geom.name = "table"
    table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    table_geom.size = [TABLE_HALF[0], TABLE_HALF[1], TABLE_THICK / 2.0]
    table_geom.pos = [TABLE_CENTRE[0], TABLE_CENTRE[1], TABLE_HEIGHT - TABLE_THICK / 2.0]
    table_geom.rgba = [0.55, 0.45, 0.35, 1.0]

    # 6. No chute. There is one flat table and nothing else -- a picked object
    #    is set back down on the tabletop. The old sunk bin was a simulation
    #    convenience that does not exist in a real cell, and as placed it sat
    #    inside the table footprint with half its scatter radius off the edge.

    # 7. Object pool: POOL_SIZE bodies, each with its own freejoint.
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

    # 7b. Calibration board (optional): slab with a grid of emissive markers.
    if with_calibration_board:
        half = 0.5 * (CALIB_MARKERS - 1) * CALIB_SPACING + CALIB_BOARD_MARGIN

        # emission is a material property in MuJoCo, not a geom one.
        slab_mat = arm.add_material()
        slab_mat.name = "calib_slab_mat"
        slab_mat.rgba = [0.02, 0.02, 0.02, 1.0]
        slab_mat.emission = 0.0

        marker_mat = arm.add_material()
        marker_mat.name = "calib_marker_mat"
        marker_mat.rgba = [1.0, 1.0, 1.0, 1.0]
        marker_mat.emission = 1.0

        board = arm.worldbody.add_body()
        board.name = "calib_board"
        board.pos = PARK_POS
        j = board.add_freejoint()
        j.name = "calib_board_free"

        slab = board.add_geom()
        slab.name = "calib_slab"
        slab.type = mujoco.mjtGeom.mjGEOM_BOX
        slab.size = [half, half, 0.5 * CALIB_BOARD_THICK]
        slab.material = "calib_slab_mat"
        slab.contype = 0
        slab.conaffinity = 0

        c = 0.5 * (CALIB_MARKERS - 1)
        for i in range(CALIB_MARKERS):
            for k in range(CALIB_MARKERS):
                g = board.add_geom()
                g.name = f"calib_m_{i}_{k}"
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [CALIB_MARKER_R, 0.0, 0.0]
                # local marker grid: XY plane of the board, front face at +z.
                g.pos = [(i - c) * CALIB_SPACING, (k - c) * CALIB_SPACING,
                         0.5 * CALIB_BOARD_THICK]
                g.material = "calib_marker_mat"
                g.contype = 0
                g.conaffinity = 0

    # 8. Gravity is left ON for every body, including the arm.
    #
    # It used to be switched off here (body.gravcomp = 1.0 on every non-object
    # body) and that was wrong twice over. First, it made the simulated plant
    # weightless: a real UR3e carries its own weight, and the point of the model
    # is to reproduce the load the controller has to hold. Second, the backend
    # ALSO feeds g(q) forward (see robot_backend_mujoco.gravity_feedforward),
    # which is the single compensation a UR controller applies internally. With
    # both active the arm was pushed UP by g against its position servos, so a
    # commanded grasp pose sat ~7 mm high and ~2-3 mm sideways with the tool
    # tilted ~1.8 deg -- the offset that put the pads on the cube's side instead
    # of around it.
    #
    # The rule is: the model has honest gravity, and the controller emulation
    # (or the real UR controller, on hardware) compensates it exactly once.
    # Nothing to set here -- MuJoCo's default is gravcomp = 0, which is what
    # every body should keep.

    # 9. Compile
    try:
        model = arm.compile()
    except Exception as e:
        print(f"[ERROR] Failed to compile composed model: {e}")
        return None

    # 10. Gripper squeeze force limit.
    grip_act = model.nu - 1
    model.actuator_forcerange[grip_act] = GRIP_FORCE_RANGE
    print(f"[INFO] gripper force range = {GRIP_FORCE_RANGE} N")

    return model
