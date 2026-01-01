#!/usr/bin/env python3
import os
import tyro
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from lerobot.envs.sapien_env import SO101TaskEnv
from lerobot.common.camera import apply_distortion, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY

# ---------------- Main ----------------
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Args:
        headless: bool = True
        task: str = "default"  # default, lift, sort, stack

    args = tyro.cli(Args)

    valid_tasks = ["default", "lift", "sort", "stack"]
    if args.task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}")

    # 创建环境实例 - BaseEnv 需要通过 render_mode 参数控制渲染模式
    render_mode = None if args.headless else "human"
    env = SO101TaskEnv(render_mode=render_mode, robot_init_qpos_noise=0.02)
    
    # 初始化环境（必须调用）
    env.reset()
    
    scene = env.scene
    left_wrist_cam = env.left_wrist_cam
    right_wrist_cam = env.right_wrist_cam
    
    # 从 _sensors 中获取摄像头（BaseEnv 内部使用 _sensors 存储）
    sensors = getattr(env, '_sensors', {})
    front_cam = sensors.get("front_camera") if sensors else None
    world_cam = sensors.get("world_demo_camera") if sensors else None    # headless 模式下渲染和保存图像
    if args.headless:
        # 让物理仿真稳定 - 直接步进scene，避免调用env.step()触发reward计算
        for _ in range(60):
            scene.step()
        scene.update_render()

        os.makedirs(f"logs/simulation/{args.task}", exist_ok=True)

    # 前置摄像头 (ManiSkill Camera 对象 - 使用 camera 属性访问底层 SAPIEN 摄像头)
        if front_cam:
            try:
                rgba = None
                
                # 直接访问底层 SAPIEN 摄像头
                if hasattr(front_cam, 'camera'):
                    try:
                        cam = front_cam.camera
                        if hasattr(cam, 'take_picture'):
                            cam.take_picture()
                            rgba = cam.get_picture("Color")
                    except Exception as e:
                        pass
                
                # 备选：尝试 capture 无参数版本
                if rgba is None and hasattr(front_cam, 'capture'):
                    try:
                        front_cam.capture()
                        if hasattr(front_cam, 'camera'):
                            cam = front_cam.camera
                            if hasattr(cam, 'take_picture'):
                                cam.take_picture()
                                rgba = cam.get_picture("Color")
                    except Exception as e:
                        pass
                
                if rgba is not None:
                    # 第一步：处理列表/元组
                    if isinstance(rgba, (list, tuple)):
                        rgba = rgba[0] if len(rgba) > 0 else None
                    
                    # 第二步：转换 Tensor 到 numpy（在提取元素之后）
                    if rgba is not None and hasattr(rgba, 'numpy'):
                        rgba = rgba.cpu().numpy()
                    
                    # 第三步：处理多余维度
                    if rgba is not None:
                        while len(rgba.shape) > 3:
                            rgba = rgba.squeeze(0)
                        if rgba.shape[0] == 1:
                            rgba = rgba.squeeze(0)
                        
                        # 转换为 uint8 [0, 255]
                        front_img = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
                        # 应用畸变
                        front_img = apply_distortion(front_img, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY)
                        Image.fromarray(front_img).save(os.path.join(f"logs/simulation/{args.task}", "front.png"))
                        print(f"Saved front camera image")
            except Exception as e:
                print(f"[WARN] Failed to save front camera image: {e}")

        # 左腕部摄像头 (RenderCameraComponent or RenderCamera)
        if left_wrist_cam:
            try:
                left_rgba = None
                # 如果是 RenderCameraComponent
                if hasattr(left_wrist_cam, 'get_picture'):
                    left_wrist_cam.take_picture()
                    left_rgba = left_wrist_cam.get_picture("Color")
                # 如果是 RenderCamera，获取图像
                elif hasattr(left_wrist_cam, 'get_images'):
                    images_dict = left_wrist_cam.get_images()
                    left_rgba = images_dict['Color']
                
                if left_rgba is not None:
                    # 第一步：处理列表/元组（get_images 可能返回列表格式）
                    if isinstance(left_rgba, (list, tuple)):
                        left_rgba = left_rgba[0] if len(left_rgba) > 0 else None
                    
                    # 第二步：转换 Tensor 到 numpy（在提取元素之后）
                    if left_rgba is not None and hasattr(left_rgba, 'numpy'):
                        left_rgba = left_rgba.cpu().numpy()
                    
                    # 第三步：使用转换后的数据
                    if left_rgba is not None:
                        # 处理多余维度 - 如果有批次维度就压缩掉
                        while len(left_rgba.shape) > 3:
                            left_rgba = left_rgba.squeeze(0)
                        
                        # 确保形状是 (H, W, C)
                        if left_rgba.shape[0] == 1:
                            left_rgba = left_rgba.squeeze(0)
                        
                        left_img = (left_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(left_img).save(os.path.join(f"logs/simulation/{args.task}", "left_wrist.png"))
                        print(f"Saved left wrist camera image")
            except Exception as e:
                print(f"[WARN] Failed to save left wrist camera image: {e}")

        # 右腕部摄像头
        if right_wrist_cam:
            try:
                right_rgba = None
                # 如果是 RenderCameraComponent
                if hasattr(right_wrist_cam, 'get_picture'):
                    right_wrist_cam.take_picture()
                    right_rgba = right_wrist_cam.get_picture("Color")
                # 如果是 RenderCamera，获取图像
                elif hasattr(right_wrist_cam, 'get_images'):
                    images_dict = right_wrist_cam.get_images()
                    right_rgba = images_dict['Color']
                
                if right_rgba is not None:
                    # 第一步：处理列表/元组（get_images 可能返回列表格式）
                    if isinstance(right_rgba, (list, tuple)):
                        right_rgba = right_rgba[0] if len(right_rgba) > 0 else None
                    
                    # 第二步：转换 Tensor 到 numpy（在提取元素之后）
                    if right_rgba is not None and hasattr(right_rgba, 'numpy'):
                        right_rgba = right_rgba.cpu().numpy()
                    
                    # 第三步：使用转换后的数据
                    if right_rgba is not None:
                        # 处理多余维度 - 如果有批次维度就压缩掉
                        while len(right_rgba.shape) > 3:
                            right_rgba = right_rgba.squeeze(0)
                        
                        # 确保形状是 (H, W, C)
                        if right_rgba.shape[0] == 1:
                            right_rgba = right_rgba.squeeze(0)
                        
                        right_img = (right_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(right_img).save(os.path.join(f"logs/simulation/{args.task}", "right_wrist.png"))
                        print(f"Saved right wrist camera image")
            except Exception as e:
                print(f"[WARN] Failed to save right wrist camera image: {e}")

        # 世界摄像头（演示用）
        if world_cam:
            try:
                demo_rgba = None
                
                # 直接访问底层 SAPIEN 摄像头
                if hasattr(world_cam, 'camera'):
                    try:
                        cam = world_cam.camera
                        if hasattr(cam, 'take_picture'):
                            cam.take_picture()
                            demo_rgba = cam.get_picture("Color")
                    except Exception as e:
                        pass
                
                # 备选：尝试 capture 无参数版本
                if demo_rgba is None and hasattr(world_cam, 'capture'):
                    try:
                        world_cam.capture()
                        if hasattr(world_cam, 'camera'):
                            cam = world_cam.camera
                            if hasattr(cam, 'take_picture'):
                                cam.take_picture()
                                demo_rgba = cam.get_picture("Color")
                    except Exception as e:
                        pass
                
                if demo_rgba is not None:
                    # 第一步：处理列表/元组
                    if isinstance(demo_rgba, (list, tuple)):
                        demo_rgba = demo_rgba[0] if len(demo_rgba) > 0 else None
                    
                    # 第二步：转换 Tensor 到 numpy（在提取元素之后）
                    if demo_rgba is not None and hasattr(demo_rgba, 'numpy'):
                        demo_rgba = demo_rgba.cpu().numpy()
                    
                    # 第三步：处理多余维度
                    if demo_rgba is not None:
                        while len(demo_rgba.shape) > 3:
                            demo_rgba = demo_rgba.squeeze(0)
                        if demo_rgba.shape[0] == 1:
                            demo_rgba = demo_rgba.squeeze(0)
                        
                        demo_img = (demo_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(demo_img).save(os.path.join(f"logs/simulation/{args.task}", "demo.png"))
                        print(f"Saved world demo camera image")
            except Exception as e:
                print(f"[WARN] Failed to save world demo camera image: {e}")

        print(f"Completed rendering for task {args.task} in logs/simulation/{args.task}/")
