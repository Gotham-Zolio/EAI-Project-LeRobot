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
    loader.fix_root_link = fix_root_link
    # Note: Ensure the assets folder is in the correct path relative to this script
    try:
        robot = loader.load("reference-scripts/assets/SO101/so101.urdf")
    except Exception:
        # Fallback if running from root and assets are in reference-scripts
        robot = loader.load("assets/SO101/so101.urdf")
        
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    
    # Set initial joint positions
    arm_init_qpos = [0, 0, 0, 0, 0]
    gripper_init_qpos = [0]
    robot.set_qpos(arm_init_qpos + gripper_init_qpos)

    # 5. Camera Setup (from front_camera.py)
    camera_mount = sapien.Entity()
    camera = RenderCameraComponent(640, 480)
    camera.set_fovx(np.deg2rad(117.12), compute_y=False)
    camera.set_fovy(np.deg2rad(73.63), compute_x=False)
    camera.near = 0.01
    camera.far = 100
    camera_mount.add_component(camera)
    camera_mount.name = "front_camera"
    scene.add_entity(camera_mount)

    # Camera Pose
    cam_rot = np.array([
        [np.cos(np.pi/2), 0, np.sin(np.pi/2)],
        [0, 1, 0],
        [-np.sin(np.pi/2), 0, np.cos(np.pi/2)],
    ])
    mat44 = np.eye(4)
    mat44[:3, :3] = cam_rot
    mat44[:3, 3] = np.array([0.26, 0.316, 0.407]) # Offset from front_camera.py
    camera_mount.set_pose(sapien.Pose(mat44))

    # 6. Viewer Setup
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-1, y=0.3, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    print("Simulation started. Close the viewer window to exit.")
    
    while not viewer.closed:
        for _ in range(4):
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()

        os.makedirs("logs", exist_ok=True)

        # Save front camera image (with distortion)
        front_cam.take_picture()
        rgba = (front_cam.get_picture("Color") * 255).astype("uint8")
        rgba = apply_distortion(rgba, 570.21740069, 570.17974410, 320, 240)
        Image.fromarray(rgba).save("logs/front_camera.png")

        print("Saved logs/front_camera.png")
