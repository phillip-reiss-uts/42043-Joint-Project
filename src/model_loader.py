import mujoco
import mujoco_menagerie as mm

# Camera position and rotation
# Position: 5 cm off-axis, 3 cm back along -Z.
CAM_QUAT = [0.0, 0.707, 0.707, 0.0]
CAM_POS = [0.05, 0.0, -0.03]

# Camer mesh position and rotation
MESH_QUAT = [0.707, 0.0, 0.0, 0.707]
MESH_POS = CAM_POS #[0.05, 0.0, -0.03]

def load_robot_model():
    """
    Compose UR5e + Robotiq 2F-85 + Realsense D435i + ground plane.
    Adds a functional MuJoCo camera element to the gripper base so the
    wrist camera can actually render RGB-D.

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
    gripper_frame.pos  = MESH_POS
    gripper_frame.quat = MESH_QUAT#[1.0, 0.0, 0.0, 0.0]  # no rotation on the frame

    arm.attach(camera, prefix="camera/", frame=gripper_frame)

    # ------------------------------------------------------------------
    # 3. Add the camera element to the gripper base.
    #    Must be after attach: MjSpec.attach() does not propagate
    #    <camera> elements from the child spec into the parent.
    # ------------------------------------------------------------------
    gripper_base.add_camera(
        name="depth_cam",
        pos=CAM_POS,
        quat=CAM_QUAT,
        fovy=60.0,
    )
    print(f"[INFO] Added 'depth_cam' to {gripper_base.name}")

    # ------------------------------------------------------------------
    # 4. Ground plane with checkerboard texture
    # ------------------------------------------------------------------
    floor_geom = arm.worldbody.add_geom()
    floor_geom.name = "floor"
    floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor_geom.size = [10, 10, 0.1]
    floor_geom.pos = [0, 0, 0]

    try:
        texture = arm.add_texture()
        texture.name = "groundplane"
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        texture.rgb1 = [0.2, 0.3, 0.4]
        texture.rgb2 = [0.1, 0.2, 0.3]
        texture.width = 300
        texture.height = 300

        material = arm.add_material()
        material.name = "groundplane"
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        material.texuniform = True
        material.texrepeat = [10, 10]

        floor_geom.material = "groundplane"
    except Exception as e:
        print(f"[WARNING] Could not add floor texture: {e}")

    # ------------------------------------------------------------------
    # 5. Compile
    # ------------------------------------------------------------------

    try:
        model = arm.compile()
    except Exception as e:
        print(f"[ERROR] Failed to compile composed model: {e}")
        return None

    return model