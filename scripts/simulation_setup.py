#!/usr/bin/env python3
import os
import itertools

import numpy as np
import cv2
import tyro
import sapien
from sapien.pysapien.render import RenderCameraComponent
from sapien.core import Entity, Pose
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

# ---------------- Global Constants ----------------
CM = 0.01

# Front camera intrinsics
FRONT_CAM_W = 640
FRONT_CAM_H = 480
FRONT_FX = 570.21740069
FRONT_FY = 570.17974410
FRONT_CX = FRONT_CAM_W / 2
FRONT_CY = FRONT_CAM_H / 2

# Distortion coefficients (for applying to front camera image)
K1 = -0.735413911
K2 = 0.949258417
P1 = 0.000189059
P2 = -0.002003513
K3 = -0.864150312


# ---------------- Utility / Scene building ----------------
def add_box(scene, center, size, color):
    """Add a thin box (used for boundary lines)."""
    actor_builder = scene.create_actor_builder()
    half = [s / 2 for s in size]
    material = sapien.render.RenderMaterial()
    material.base_color = np.array(color)
    actor_builder.add_box_collision(half_size=half)
    actor_builder.add_box_visual(half_size=half, material=material)
    actor = actor_builder.build()
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

    floor = builder.build()
    floor.set_pose(sapien.Pose([width/2, height/2, -thickness/2]))
    return floor


def load_arm(scene, urdf_path, root_x):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    arm = loader.load(urdf_path)

    quat = R.from_euler('xyz', [0.0, 0.0, np.pi / 2]).as_quat()
    quat_sapien = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

    temp_pose = sapien.Pose(p=[root_x, 0.0, 0.0], q=quat_sapien)
    arm.set_root_pose(temp_pose)

    aabb_min, _ = arm.get_links()[0].get_global_aabb_fast()
    root_y = -aabb_min[1]

    pose = sapien.Pose(p=[root_x, root_y, 0.0], q=quat_sapien)
    arm.set_root_pose(pose)

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
    # attach camera component to the link's entity
    link.entity.add_component(cam)

    # local pose: lift it a bit along link's local z and rotate to look down
    offset = np.array([0.0, 0.0, z_offset], dtype=np.float32)
    quat = R.from_euler('xyz', [-np.pi/2, 0.0, 0.0]).as_quat()  # xyzw
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]         # wxyz

    cam.set_local_pose(sapien.Pose(offset, quat_sapien))
    return cam


def apply_distortion(img, fx, fy, cx, cy, k1=K1, k2=K2, p1=P1, p2=P2, k3=K3):
    """
    Apply radial + tangential distortion to an image (numpy array HxWxC uint8).
    fx,fy,cx,cy should match the camera intrinsics used to render the image.
    """
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xd = (xs - cx) / fx
    yd = (ys - cy) / fy
    r2 = xd * xd + yd * yd
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x = (xd - 2 * p1 * xd * yd - p2 * (r2 + 2 * xd * xd)) / radial
    y = (yd - p1 * (r2 + 2 * yd * yd) - 2 * p2 * xd * yd) / radial
    u = (x * fx + cx).astype(np.float32)
    v = (y * fy + cy).astype(np.float32)

    distorted_img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return distorted_img


def add_block(scene, center, color, label="A"):
    """
    Add a colored block to the scene.
    center: [x,y,z] in meters (world frame)
    color: [r,g,b,a] values in 0..1
    label: a short text label (drawn onto a PIL image; SAPIEN 3.0 does not support set_texture here)
    """
    size = [3 * CM, 3 * CM, 3 * CM]

    actor_builder = scene.create_actor_builder()
    half = [s / 2 for s in size]
    material = sapien.render.RenderMaterial()
    material.base_color = np.array(color)
    actor_builder.add_box_collision(half_size=half)
    actor_builder.add_box_visual(half_size=half, material=material)

    actor = actor_builder.build()

    # create a small label image (we keep this for reference; applying to visual requires API not present)
    tex_size = 256
    img = Image.new("RGBA", (tex_size, tex_size),
                    (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((tex_size - w) / 2, (tex_size - h) / 2), label, fill=(255, 255, 255, 255), font=font)
    tex_np = np.array(img).astype(np.float32) / 255.0

    # Note: SAPIEN 3.x RenderVisual.set_texture may not exist in your build.
    # If you have an API to set texture, you can apply `tex_np` to the visual here.
    # visual = actor.get_render_body().visuals[0]
    # visual.set_texture(tex_np)

    actor.set_pose(sapien.Pose(center))
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
    urdf_path = "reference-scripts/assets/SO101/so101.urdf"
    left_arm = load_arm(scene, urdf_path, root_x=11.9 * CM)
    right_arm = load_arm(scene, urdf_path, root_x=48.1 * CM)

    # ------ Wrist cameras ------
    left_wrist_cam = add_wrist_camera(left_arm, link_name="camera_link", fovy_deg=70.0, z_offset=0.07)
    right_wrist_cam = add_wrist_camera(right_arm, link_name="camera_link", fovy_deg=70.0, z_offset=0.07)

    return scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam


# ---------------- Scene configurations (three tasks) ----------------
def setup_scene_1(scene):
    """Scene 1: one red block in the leftmost box"""
    add_block(scene, center=[12 * CM, 25 * CM, 1.5 * CM], color=[1.0, 0.0, 0.0, 1.0], label="Red")


def setup_scene_2(scene):
    """Scene 2: red + green in leftmost box (different locs)"""
    add_block(scene, center=[8 * CM, 22 * CM, 1.5 * CM], color=[1.0, 0.0, 0.0, 1.0], label="R")
    add_block(scene, center=[16 * CM, 28 * CM, 1.5 * CM], color=[0.0, 0.8, 0.0, 1.0], label="G")


def setup_scene_3(scene):
    """Scene 3: red + green in the middle box"""
    add_block(scene, center=[27 * CM, 23 * CM, 1.5 * CM], color=[1.0, 0.0, 0.0, 1.0], label="R")
    add_block(scene, center=[33 * CM, 27 * CM, 1.5 * CM], color=[0.0, 0.8, 0.0, 1.0], label="G")


# ---------------- Main ----------------
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Args:
        headless: bool = True
        scene: int = 0  # 0,1,2,3

    args = tyro.cli(Args)

    if args.scene not in [0, 1, 2, 3]:
        raise ValueError("scene must be 0, 1, 2 or 3")

    scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam = create_scene(headless=args.headless)

    # populate blocks according to requested scene
    if args.scene == 1:
        setup_scene_1(scene)
    elif args.scene == 2:
        setup_scene_2(scene)
    elif args.scene == 3:
        setup_scene_3(scene)

    # headless rendering / save images
    if args.headless:
        # step a bit to let physics settle
        for _ in range(60):
            scene.step()
        scene.update_render()

        os.makedirs("logs", exist_ok=True)

        # front camera
        front_cam.take_picture()
        rgba = (front_cam.get_picture("Color") * 255).astype("uint8")
        rgba = apply_distortion(rgba, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY)
        Image.fromarray(rgba).save(os.path.join("logs", f"front_camera_scene{args.scene}.png"))

        # wrist cameras
        left_wrist_cam.take_picture()
        lw_img = (left_wrist_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(lw_img).save(os.path.join("logs", f"left_wrist_camera_scene{args.scene}.png"))

        right_wrist_cam.take_picture()
        rw_img = (right_wrist_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(rw_img).save(os.path.join("logs", f"right_wrist_camera_scene{args.scene}.png"))

        print(f"Saved front and wrist camera images for scene {args.scene} in logs/")
