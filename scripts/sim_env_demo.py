#!/usr/bin/env python3
import os
import tyro
from PIL import Image
import numpy as np

from lerobot.envs.sapien_env import create_scene, setup_scene_1, setup_scene_2, setup_scene_3
from lerobot.common.camera import apply_distortion, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY

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

        os.makedirs("logs/simulation/captures", exist_ok=True)

        # front camera
        front_cam.take_picture()
        rgba = (front_cam.get_picture("Color") * 255).astype("uint8")
        rgba = apply_distortion(rgba, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY)
        Image.fromarray(rgba).save(os.path.join("logs/simulation/captures", f"front_camera_scene{args.scene}.png"))

        # wrist cameras
        left_wrist_cam.take_picture()
        lw_img = (left_wrist_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(lw_img).save(os.path.join("logs/simulation/captures", f"left_wrist_camera_scene{args.scene}.png"))

        right_wrist_cam.take_picture()
        rw_img = (right_wrist_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(rw_img).save(os.path.join("logs/simulation/captures", f"right_wrist_camera_scene{args.scene}.png"))

        print(f"Saved front and wrist camera images for scene {args.scene} in logs/simulation/captures/")
