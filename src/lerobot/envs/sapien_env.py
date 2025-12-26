import numpy as np
import sapien
from sapien.pysapien.render import RenderCameraComponent, RenderTexture2D
from sapien import Entity, Pose
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R
from lerobot.common.camera import (
    FRONT_CAM_W, FRONT_CAM_H, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY
)

# ---------------- Global Constants ----------------
CM = 0.01

# ---------------- Utility / Scene building ----------------
def add_box(scene, center, size, color):
    """Add a thin box (used for boundary lines)."""
    actor_builder = scene.create_actor_builder()
    half = [s / 2 for s in size]
    material = sapien.render.RenderMaterial()
    material.base_color = np.array(color)
    actor_builder.add_box_collision(half_size=half)
    actor_builder.add_box_visual(half_size=half, material=material)
    actor = actor_builder.build_static()
    actor.set_pose(sapien.Pose(center))
    return actor


def add_floor(scene):
    width = 120 * CM
    height = 60 * CM
    thickness = 0.01

    builder = scene.create_actor_builder()
    half = [width / 2, height / 2, thickness / 2]

    material = sapien.render.RenderMaterial()
    material.base_color = np.array([0.92, 0.92, 0.92, 1])
    material.specular = 0.1

    builder.add_box_collision(half_size=half)
    builder.add_box_visual(half_size=half, material=material)

    floor = builder.build_static()
    floor.set_pose(sapien.Pose([width/2, height/2, -thickness/2]))
    return floor


def load_arm(scene, urdf_path, root_x):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    arm = loader.load(urdf_path)

    # Set drive properties for position control
    for joint in arm.get_active_joints():
        joint.set_drive_property(stiffness=400, damping=40, force_limit=3.0)

    quat = R.from_euler('xyz', [0.0, 0.0, np.pi / 2]).as_quat()
    quat_sapien = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

    arm.set_root_pose(sapien.Pose([root_x, 0.0, 0.0], quat_sapien))

    base_link = arm.get_links()[0]
    aabb_min, aabb_max = base_link.get_global_aabb_fast()
    base_width_y = aabb_max[1] - aabb_min[1]

    root_y = -aabb_max[1] + base_width_y
    arm.set_root_pose(sapien.Pose([root_x, root_y, 0.0], quat_sapien))

    active_joints = arm.get_active_joints()
    n = len(active_joints)
    ready_qpos = np.zeros(n, dtype=np.float32)
    ready_qpos[:6] = np.array([0.0, -0.4, 0.2, 2.0, 0.0, 0.3], dtype=np.float32)
    arm.set_qpos(ready_qpos)

    for i, joint in enumerate(active_joints):
        joint.set_drive_target(ready_qpos[i])

    return arm


def add_wrist_camera(robot, link_name="camera_link", fovy_deg=50.0, z_offset=0.05, near=0.01, far=5.0):
    """
    Attach a RenderCameraComponent to a link (wrist).
    """
    link = robot.find_link_by_name(link_name)
    if link is None:
        raise ValueError(f"Link named '{link_name}' not found on robot")

    cam_w = 640
    cam_h = 480
    fovy = np.deg2rad(fovy_deg)
    fx = cam_w / (2 * np.tan(fovy / 2))
    fy = fx
    cx = cam_w / 2
    cy = cam_h / 2

    cam = RenderCameraComponent(width=cam_w, height=cam_h)
    cam.set_perspective_parameters(near, far, fx, fy, cx, cy, skew=0.0)

    link.entity.add_component(cam)

    offset = np.array([0.0, 0.0, z_offset], dtype=np.float32)
    quat = R.from_euler('xyz', [-np.pi/2, 0.0, 0.0]).as_quat()  # xyzw
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]         # wxyz

    cam.set_local_pose(sapien.Pose(offset, quat_sapien))
    return cam


def add_block(scene, center, color, label="A", rotation_z=0.0):
    """
    Add a colored block with a number shown ONLY on the top face.
    """
    size = np.array([3 * CM, 3 * CM, 3 * CM], dtype=np.float32)
    half = size / 2

    # ===============================
    # 1. Base cube (pure color)
    # ===============================
    base_material = sapien.render.RenderMaterial()
    base_material.base_color = np.array(color)
    base_material.roughness = 0.6
    base_material.specular = 0.2

    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half.tolist())
    builder.add_box_visual(half_size=half.tolist(), material=base_material)

    actor = builder.build()
    actor.name = f"block_{label}"

    quat = R.from_euler('z', rotation_z).as_quat()
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    actor.set_pose(sapien.Pose(center, quat_sapien))



    # ===============================
    # Create number texture (PIL)
    # ===============================
    tex_size = 256
    img = Image.new(
        "RGBA",
        (tex_size, tex_size),
        (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255)
    )
    draw = ImageDraw.Draw(img)

    font_size = 60
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size
        )
    except IOError:
        font = ImageFont.load_default()

    text = label

    # ---- 测量文字 ----
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # ---- 从“左边开始写”，但整体居中 ----
    margin_x = (tex_size - text_w) // 2 - 30
    margin_y = (tex_size - text_h) // 2

    draw.text(
        (margin_x, margin_y),
        text,
        fill=(255, 255, 255, 255),
        font=font,
        anchor="lt"
    )

    # OpenGL UV fix
    img = img.transpose(Image.FLIP_TOP_BOTTOM)



    tex_data = np.array(img, dtype=np.uint8)
    try:
        texture = RenderTexture2D(tex_data, format="R8G8B8A8Unorm")
    except ValueError:
        texture = RenderTexture2D(tex_data, format="r8g8b8a8unorm")

    num_material = sapien.render.RenderMaterial()
    num_material.base_color = np.array([1, 1, 1, 1])
    num_material.diffuse_texture = texture
    num_material.roughness = 0.4
    num_material.specular = 0.1

    # ===============================
    # 3. Number plane (贴在上表面)
    # ===============================
    plane_builder = scene.create_actor_builder()

    plane_size = [half[0]*0.9, half[1]*0.9, 0.0005]  # very thin
    plane_builder.add_box_visual(
        half_size=plane_size,
        material=num_material
    )

    plane = plane_builder.build_static()

    # small offset to avoid z-fighting
    offset_z = half[2] + 0.0006
    plane_pose = sapien.Pose(
        [center[0], center[1], center[2] + offset_z],
        quat_sapien
    )
    plane.set_pose(plane_pose)

    return actor



# ---------------- Scene assembly ----------------
def create_scene(fix_root_link: bool = True, balance_passive_force: bool = True, headless: bool = False):
    """
    Create the scene, robots and cameras.
    Returns: scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam
    """
    # ------ Scene setup ------
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0.3, 1, -0.3], [0.7, 0.7, 0.7])
    scene.add_directional_light([-0.3, 1, -0.1], [0.4, 0.4, 0.4])

    # ------ Floor (Desk) ------
    add_floor(scene)

    # ------ Boundary lines ------
    BORDER = 1.8 * CM
    d = 0.01 * CM
    black = [0, 0, 0, 1]

    # Boundaries defining the 3 boxes
    add_box(scene, center=[60 * CM, 30 * CM, d / 2], size=[BORDER, 60 * CM, d], color=black)
    add_box(scene, center=[2.9 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[21.3 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[38.7 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[57.1 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[21.3 * CM, 7.5 * CM, d / 2], size=[BORDER, 15 * CM, d], color=black)
    add_box(scene, center=[38.7 * CM, 7.5 * CM, d / 2], size=[BORDER, 15 * CM, d], color=black)
    add_box(scene, center=[30 * CM, 15.9 * CM, d / 2], size=[56 * CM, BORDER, d], color=black)
    add_box(scene, center=[30 * CM, 34.1 * CM, d / 2], size=[56 * CM, BORDER, d], color=black)

    # ------ Front camera ------
    cam_mount = Entity()
    front_cam = RenderCameraComponent(width=FRONT_CAM_W, height=FRONT_CAM_H)
    near, far = 0.01, 50.0
    front_cam.set_perspective_parameters(near, far, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY, skew=0.0)
    cam_mount.add_component(front_cam)

    cam_x, cam_y, cam_z = 31.6 * CM, 26.0 * CM, 40.7 * CM
    quat = R.from_euler('xyz', [0.0, np.pi / 2, -np.pi / 2]).as_quat()
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    cam_mount.set_pose(Pose([cam_x, cam_y, cam_z], quat_sapien))
    scene.add_entity(cam_mount)

    # ------ Robot arms ------
    urdf_path = "assets/SO101/so101.urdf"
    left_arm = load_arm(scene, urdf_path, root_x=11.9 * CM)
    right_arm = load_arm(scene, urdf_path, root_x=48.1 * CM)

    for arm in [left_arm, right_arm]:
        qpos = arm.get_qpos()
        for i, joint in enumerate(arm.get_active_joints()):
            joint.set_drive_property(stiffness=400, damping=40, force_limit=3.0)
            joint.set_drive_target(qpos[i])
    
    for _ in range(100):
        scene.step()
        
    for arm in [left_arm, right_arm]:
        qpos = arm.get_qpos()
        for i, joint in enumerate(arm.get_active_joints()):
            joint.set_drive_property(stiffness=400, damping=40, force_limit=3.0)
            joint.set_drive_target(qpos[i])

    # ------ Wrist cameras ------
    left_wrist_cam = add_wrist_camera(left_arm, link_name="camera_link", fovy_deg=70.0, z_offset=0.07)
    right_wrist_cam = add_wrist_camera(right_arm, link_name="camera_link", fovy_deg=70.0, z_offset=0.07)

    # ------ World demo camera ------
    world_cam_mount = Entity()
    world_cam = RenderCameraComponent(width=FRONT_CAM_W, height=FRONT_CAM_H)
    near, far = 0.01, 50.0
    world_cam.set_perspective_parameters(near, far, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY, skew=0.0)
    world_cam_mount.add_component(world_cam)

    cam_x, cam_y, cam_z = -14.0 * CM, 60.0 * CM, 40.0 * CM
    quat = R.from_euler('xyz', [0.0, np.pi / 6, -np.pi / 4]).as_quat()
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    world_cam_mount.set_pose(Pose([cam_x, cam_y, cam_z], quat_sapien))
    scene.add_entity(world_cam_mount)

    return scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam, world_cam


# ---------------- Scene configurations ----------------
def get_random_pose(x_range, y_range, z_height):
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    z = z_height
    rot_z = np.random.uniform(0, 2 * np.pi)
    return [x, y, z], rot_z

def setup_scene_lift(scene):
    """Task Lift: one red block in the rightmost box"""
    pos, rot = get_random_pose([44.0 * CM, 47.0 * CM], [22.0 * CM, 24.5 * CM], 1.5 * CM)
    actor = add_block(scene, center=pos, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot)
    return [actor]


def setup_scene_stack(scene):
    """Task Sort: red + green in rightmost box"""
    while True:
        pos1, rot1 = get_random_pose([41.1 * CM, 54.7 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM)
        pos2, rot2 = get_random_pose([41.1 * CM, 54.7 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM)
        dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
        if dist >= 4.5 * CM:
            break
            
    a1 = add_block(scene, center=pos1, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot1)
    a2 = add_block(scene, center=pos2, color=[0.0, 0.8, 0.0, 1.0], label="Green", rotation_z=rot2)
    return [a1, a2]


def setup_scene_sort(scene):
    """Task Stack: red + green in the middle box"""
    while True:
        pos1, rot1 = get_random_pose([23.7 * CM, 36.3 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM)
        pos2, rot2 = get_random_pose([23.7 * CM, 36.3 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM)
        dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
        if dist >= 4.5 * CM:
            break

    a1 = add_block(scene, center=pos1, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot1)
    a2 = add_block(scene, center=pos2, color=[0.0, 0.8, 0.0, 1.0], label="Green", rotation_z=rot2)
    return [a1, a2]

def setup_scene_operation(scene):
    """
    Task Operation:
    1. Left Box: Random block 0-9.
    2. Right Box: Random block 0-9.
    3. Middle Box: 3 blocks.
       - One is (left + right) % 10.
       - Two are random unique distractors (0-9).
    All blocks are Red with White text.
    """
    actors = []
    
    # --- 1. Logic ---
    val_l = np.random.randint(0, 10) 
    val_r = np.random.randint(0, 10) 
    
    correct_ans = (val_l + val_r) % 10
    
    # Generate distractors: 0-9, excluding the correct answer
    candidates = list(range(10))
    candidates.remove(correct_ans)
    distractors = np.random.choice(candidates, 2, replace=False)
    
    # Middle values
    middle_values = [correct_ans, distractors[0], distractors[1]]
    np.random.shuffle(middle_values)
    
    # Color definition (Red background)
    block_color = [1.0, 0.0, 0.0, 1.0]
    
    # --- 2. Placement Ranges ---
    # Safe Y range
    range_y = [18.5 * CM, 30.0 * CM]
    
    # Left Box X safe range: [6.0, 18.0]
    pos_l, rot_l = get_random_pose([6.0 * CM, 18.0 * CM], range_y, 1.5 * CM)
    actors.append(add_block(scene, pos_l, block_color, str(val_l), rot_l))
    
    # Right Box X safe range: [42.0, 54.0]
    pos_r, rot_r = get_random_pose([42.0 * CM, 54.0 * CM], range_y, 1.5 * CM)
    actors.append(add_block(scene, pos_r, block_color, str(val_r), rot_r))
    
    # Middle Box X safe range: [24.0, 36.0]
    mid_positions = []
    
    for val in middle_values:
        while True:
            pos, rot = get_random_pose([24.0 * CM, 36.0 * CM], range_y, 1.5 * CM)
            
            # Distance check
            collision = False
            for existing_p in mid_positions:
                dist = np.linalg.norm(np.array(pos[:2]) - np.array(existing_p[:2]))
                if dist < 4.5 * CM: # Check for overlap
                    collision = True
                    break
            
            if not collision:
                mid_positions.append(pos)
                actors.append(add_block(scene, pos, block_color, str(val), rot))
                break
                
    return actors

def setup_scene(scene, task_name):
    if task_name == "default":
        return []
    elif task_name == "lift":
        return setup_scene_lift(scene)
    elif task_name == "sort":
        return setup_scene_sort(scene)
    elif task_name == "stack":
        return setup_scene_stack(scene)
    elif task_name == "operation":
        return setup_scene_operation(scene)
    else:
        raise ValueError(f"Unknown task: {task_name}")