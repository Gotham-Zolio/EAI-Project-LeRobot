import sapien
import numpy as np
import tyro
from sapien.pysapien.render import RenderCameraComponent
from sapien.core import Entity, Pose

import os
from PIL import Image, ImageDraw, ImageFont
import itertools
from scipy.spatial.transform import Rotation as R
import cv2

CM = 0.01


# ------------------------ Utils ------------------------
def add_box(scene, center, size, color):
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


# ------------------------ Robot ------------------------
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


def add_wrist_camera(robot, link_name="camera_link"):
    link = robot.find_link_by_name(link_name)

    cam_w = 640
    cam_h = 480
    fovy = np.deg2rad(50)
    fx = cam_w / (2 * np.tan(fovy / 2))
    fy = fx
    cx = cam_w / 2
    cy = cam_h / 2
    near = 0.01
    far = 5.0

    cam = RenderCameraComponent(width=cam_w, height=cam_h)
    cam.set_perspective_parameters(near, far, fx, fy, cx, cy, skew=0.0)
    link.entity.add_component(cam)
    cam.set_entity_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    return cam


# ------------------------ Camera Distortion ------------------------
def apply_distortion(img, fx, fy, cx, cy):
    h, w = img.shape[:2]

    k1, k2, p1, p2, k3 = [
        -0.735413911,
         0.949258417,
         0.000189059,
        -0.002003513,
        -0.864150312
    ]

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xd = (xs - cx) / fx
    yd = (ys - cy) / fy
    r2 = xd*xd + yd*yd
    r4 = r2*r2
    r6 = r4*r2

    radial = 1 + k1*r2 + k2*r4 + k3*r6
    x = (xd - 2*p1*xd*yd - p2*(r2 + 2*xd*xd)) / radial
    y = (yd - p1*(r2 + 2*yd*yd) - 2*p2*xd*yd) / radial
    u = (x * fx + cx).astype(np.float32)
    v = (y * fy + cy).astype(np.float32)

    distorted_img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR)
    return distorted_img


# ------------------------ Block ------------------------
def add_block(scene, center, size=3*CM, color=[1,0,0,1], label="A"):
    half = [size/2]*3
    actor_builder = scene.create_actor_builder()

    # collision + visual
    actor_builder.add_box_collision(half_size=half)
    material = sapien.render.RenderMaterial()
    material.base_color = np.array(color)
    actor_builder.add_box_visual(half_size=half, material=material)

    actor = actor_builder.build()

    tex_size = 256
    img = Image.new("RGBA", (tex_size, tex_size),
                    (int(color[0]*255), int(color[1]*255), int(color[2]*255), 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = draw.textsize(label, font=font)
    draw.text(((tex_size-w)/2, (tex_size-h)/2), label,
               fill=(255,255,255,255), font=font)
    tex_np = np.array(img).astype(np.float32)/255.0

    actor.get_render_body().visuals[0].set_texture(tex_np)

    actor.set_pose(sapien.Pose(center))
    return actor


# ------------------------ Scene Construction ------------------------
def create_scene(
    fix_root_link: bool = True,
    balance_passive_force: bool = True,
    headless: bool = False,
):
    # ------ Scene init ------
    scene = sapien.Scene()
    scene.set_timestep(1/240)
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0.3, 1, -0.3], [0.7, 0.7, 0.7])
    scene.add_directional_light([-0.3, 1, -0.1], [0.4, 0.4, 0.4])

    # Floor
    add_floor(scene)

    # Boundary lines
    BORDER = 1.8 * CM
    d = 0.01 * CM
    black = [0, 0, 0, 1]

    add_box(scene, center=[60*CM,30*CM,d/2], size=[BORDER,60*CM,d], color=black)
    add_box(scene, center=[2.9*CM,25*CM,d/2], size=[BORDER,16.4*CM,d], color=black)
    add_box(scene, center=[21.3*CM,25*CM,d/2], size=[BORDER,16.4*CM,d], color=black)
    add_box(scene, center=[38.7*CM,25*CM,d/2], size=[BORDER,16.4*CM,d], color=black)
    add_box(scene, center=[57.1*CM,25*CM,d/2], size=[BORDER,16.4*CM,d], color=black)
    add_box(scene, center=[21.3*CM,7.5*CM,d/2], size=[BORDER,15*CM,d], color=black)
    add_box(scene, center=[38.7*CM,7.5*CM,d/2], size=[BORDER,15*CM,d], color=black)
    add_box(scene, center=[30*CM,15.9*CM,d/2], size=[56*CM,BORDER,d], color=black)
    add_box(scene, center=[30*CM,34.1*CM,d/2], size=[56*CM,BORDER,d], color=black)

    # Front camera
    cam_w, cam_h = 640, 480
    fx, fy = 570.21740069, 570.17974410
    cx, cy = cam_w/2, cam_h/2
    near, far = 0.01, 50.0

    cam_mount = Entity()
    cam = RenderCameraComponent(width=cam_w, height=cam_h)
    cam.set_perspective_parameters(near, far, fx, fy, cx, cy, skew=0.0)
    cam_mount.add_component(cam)

    cam_x, cam_y, cam_z = 31.6*CM, 26.0*CM, 20.3*CM
    quat = R.from_euler('xyz', [0.0, np.pi/2, -np.pi/2]).as_quat()
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    cam_mount.set_pose(Pose([cam_x, cam_y, cam_z], quat_sapien))
    scene.add_entity(cam_mount)

    # Load robot arms
    urdf_path = "reference-scripts/assets/SO101/so101.urdf"
    left_arm = load_arm(scene, urdf_path, root_x=11.9 * CM)
    right_arm = load_arm(scene, urdf_path, root_x=48.1 * CM)

    add_wrist_camera(left_arm)
    add_wrist_camera(right_arm)

    return scene, cam, left_arm, right_arm


# ------------------------ Main ------------------------
if __name__ == "__main__":
    class Args(tyro.conf.Suppress):
        headless: bool = True
        block_x: float = 0.30
        block_y: float = 0.20

    args = tyro.cli(Args)

    scene, front_cam, left_arm, right_arm = create_scene(headless=args.headless)

    # Add block (3cm, red, label A)
    add_block(scene, center=[args.block_x, args.block_y, 1.5*CM])

    if args.headless:
        for _ in range(60):
            scene.step()
        scene.update_render()

        os.makedirs("logs", exist_ok=True)

        # Save front camera image (with distortion)
        front_cam.take_picture()
        rgba = (front_cam.get_picture("Color") * 255).astype("uint8")
        rgba = apply_distortion(rgba, 570.21740069, 570.17974410, 320, 240)
        Image.fromarray(rgba).save("logs/front_camera.png")

        print("Saved logs/front_camera.png")
