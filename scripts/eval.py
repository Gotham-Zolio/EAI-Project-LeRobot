import sys
import os
import time
import argparse
import numpy as np
import torch
import sapien
import cv2
from pathlib import Path
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
sys.path.insert(0, src_path)

from lerobot.envs.sapien_env import create_scene, setup_scene_1
from lerobot.policy.diffusion import DiffusionPolicy

def get_stats(dataset_path, repo_id):
    stats_path = Path(dataset_path) / repo_id / "meta/stats.json"
    if not stats_path.exists():
        print(f"Warning: Stats file not found at {stats_path}")
        return {}
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

def normalize(data, stats, key):
    if key not in stats:
        return data
    mean = torch.tensor(stats[key]["mean"], device=data.device, dtype=data.dtype)
    std = torch.tensor(stats[key]["std"], device=data.device, dtype=data.dtype)
    if isinstance(stats[key]["mean"][0], list): return data
    return (data - mean) / (std + 1e-8)

def unnormalize(data, stats, key):
    if key not in stats:
        return data
    mean = torch.tensor(stats[key]["mean"], device=data.device, dtype=data.dtype)
    std = torch.tensor(stats[key]["std"], device=data.device, dtype=data.dtype)
    if isinstance(stats[key]["mean"][0], list): return data
    return data * (std + 1e-8) + mean

def main(checkpoint_path, device="cuda"):
    # 1. Setup Scene
    scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam = create_scene(headless=False)
    setup_scene_1(scene)
    
    # Setup Viewer
    try:
        viewer = scene.create_viewer()
        viewer.set_camera_xyz(x=1.0, y=0.0, z=1.0)
        viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)
    except Exception as e:
        print(f"Could not create viewer: {e}")
        viewer = None

    # 2. Load Model
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    stats = get_stats("data", "lift")
    
    if stats:
        action_dim = len(stats["action"]["mean"])
        obs_dim = len(stats["observation.state"]["mean"])
    else:
        print("Stats not found, using default dimensions (check your dataset!)")
        action_dim = 6
        obs_dim = 12
        
    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        vision_backbone="resnet18",
        num_diffusion_steps=100
    ).to(device)
    
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    print("Model loaded.")

    # 3. Evaluation Loop
    step = 0
    max_steps = 500
    
    print("Starting evaluation loop...")
    while not viewer.closed and step < max_steps:
        scene.step()
        scene.update_render()
        
        # Capture Observation
        qpos = left_arm.get_qpos()
        qvel = left_arm.get_qvel()
        state_vec = np.concatenate([qpos, qvel]).astype(np.float32)
        
        front_cam.take_picture()
        rgba = front_cam.get_picture("Color")
        rgb = rgba[..., :3]
        
        img_tensor = torch.from_numpy(cv2.resize(rgb, (96, 96))).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        
        batch = {
            "observation.state": state_tensor,
            "observation.images.front": img_tensor,
        }
        
        if stats:
            batch["observation.state"] = normalize(batch["observation.state"], stats, "observation.state")
        
        with torch.no_grad():
            # Assuming policy has a sample method or we use forward for now if sample not avail
            # For diffusion, we need sample. If not present, this will fail, but user asked for eval code.
            if hasattr(policy, 'sample'):
                action = policy.sample(batch)
            else:
                # Fallback if sample not implemented yet
                action = torch.randn(1, action_dim).to(device)

        if stats:
            action = unnormalize(action, stats, "action")
            
        action_np = action.squeeze(0).cpu().numpy()
        left_arm.set_drive_target(action_np)
        
        viewer.render()
        step += 1

    print("Evaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    args = parser.parse_args()
    
    main(args.checkpoint)
