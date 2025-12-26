#!/usr/bin/env python3
import os
import tyro
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from lerobot.envs.sapien_env import create_scene, setup_scene
from lerobot.common.camera import apply_distortion, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY

# ---------------- Main ----------------
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Args:
        headless: bool = True
        task: str = "default"  # default, lift, sort, stack

    args = tyro.cli(Args)

    valid_tasks = ["default", "lift", "sort", "stack", "operation"]
    if args.task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}")

    scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam, world_cam = create_scene(headless=args.headless)

    # populate blocks according to requested task
    setup_scene(scene, args.task)

    # headless rendering / save images
    if args.headless:
        # step a bit to let physics settle
        for _ in range(60):
            scene.step()
        scene.update_render()

        os.makedirs(f"logs/simulation/{args.task}", exist_ok=True)

        # front camera
        front_cam.take_picture()
        rgba = (front_cam.get_picture("Color") * 255).astype("uint8")
        rgba = apply_distortion(rgba, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY)
        Image.fromarray(rgba).save(os.path.join(f"logs/simulation/{args.task}", "front.png"))

        # wrist cameras
        left_wrist_cam.take_picture()
        lw_img = (left_wrist_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(lw_img).save(os.path.join(f"logs/simulation/{args.task}", "left_wrist.png"))

        right_wrist_cam.take_picture()
        rw_img = (right_wrist_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(rw_img).save(os.path.join(f"logs/simulation/{args.task}", "right_wrist.png"))

        # world camera
        world_cam.take_picture()
        demo_img = (world_cam.get_picture("Color") * 255).astype("uint8")
        Image.fromarray(demo_img).save(os.path.join(f"logs/simulation/{args.task}", "demo.png"))

        print(f"Saved front, wrist, and world camera images for task {args.task} in logs/simulation/{args.task}/")
