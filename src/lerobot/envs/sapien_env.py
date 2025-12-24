import numpy as np
import sapien
from sapien.pysapien.render import RenderCameraComponent
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
    Attach a RenderCameraComponent to a link (wrist). Use set_local_pose so it
    respects SAPIEN 3.x API (no set_world_pose for RenderCameraComponent).
    Returns the camera component.
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
    Add a colored block to the scene.
    center: [x,y,z] in meters (world frame)
    color: [r,g,b,a] values in 0..1
    label: a short text label (drawn onto a PIL image; SAPIEN 3.0 does not support set_texture here)
    rotation_z: rotation around z-axis in radians
    """
    size = [3 * CM, 3 * CM, 3 * CM]

    actor_builder = scene.create_actor_builder()
    half = [s / 2 for s in size]
    material = sapien.render.RenderMaterial()
    material.base_color = np.array(color)
    actor_builder.add_box_collision(half_size=half)
    actor_builder.add_box_visual(half_size=half, material=material)

    actor = actor_builder.build()
    actor.name = f"block_{label}"

    tex_size = 256
    img = Image.new("RGBA", (tex_size, tex_size),
                    (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((tex_size - w) / 2, (tex_size - h) / 2), label, fill=(255, 255, 255, 255), font=font)
    tex_np = np.array(img).astype(np.float32) / 255.0

    quat = R.from_euler('z', rotation_z).as_quat()
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    actor.set_pose(sapien.Pose(center, quat_sapien))
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

    cam_x, cam_y, cam_z = -14.0 * CM, 60.0 * CM, 40.0 * CM  # 调低 z，调整 y
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
    # Optimized spawn range for better arm reachability
    # X: 44~47cm (reduced from 50cm - high X values cause IK failures even with good Y)
    # Y: 22~24.5cm (safe range verified by testing)
    # Test data: X>48cm causes IK failures or growing offsets; X~44-47cm gives 12-15cm initial offset
    pos, rot = get_random_pose([44.0 * CM, 47.0 * CM], [22.0 * CM, 24.5 * CM], 1.5 * CM)
    actor = add_block(scene, center=pos, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot)
    return [actor]


def setup_scene_stack(scene):
    """Task Sort: red + green in rightmost box"""
    # x in 41.1~54.7cm, y in 18.3~31.7cm, dist >= 4.5cm
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
    # x in 23.7~36.3cm, y in 18.3~31.7cm, dist >= 4.5cm
    while True:
        pos1, rot1 = get_random_pose([23.7 * CM, 36.3 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM)
        pos2, rot2 = get_random_pose([23.7 * CM, 36.3 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM)
        dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
        if dist >= 4.5 * CM:
            break

    a1 = add_block(scene, center=pos1, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot1)
    a2 = add_block(scene, center=pos2, color=[0.0, 0.8, 0.0, 1.0], label="Green", rotation_z=rot2)
    return [a1, a2]

def setup_scene(scene, task_name):
    if task_name == "default":
        return []
    elif task_name == "lift":
        return setup_scene_lift(scene)
    elif task_name == "sort":
        return setup_scene_sort(scene)
    elif task_name == "stack":
        return setup_scene_stack(scene)
    else:
        raise ValueError(f"Unknown task: {task_name}")
