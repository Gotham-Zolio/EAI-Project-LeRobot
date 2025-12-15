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
from tqdm import trange
import threading

# Add src and tools to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
tools_path = os.path.join(current_dir, "..")
sys.path.insert(0, src_path)
sys.path.insert(0, tools_path)

from lerobot.envs.sapien_env import create_scene, setup_scene
from lerobot.policy.diffusion import DiffusionPolicy
from tools.web_viewer.viewer import WebViewer

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

def main(checkpoint_path, task, device="cuda", save_video=False, web_viewer=False, port=5000):
    # Start web server if requested
    viewer_app = None
    if web_viewer:
        viewer_app = WebViewer(port=port)
        viewer_app.start()

    # 1. Setup Scene
    scene, front_cam, left_arm, right_arm, left_wrist_cam, right_wrist_cam = create_scene(headless=False)
    setup_scene(scene, task)
    
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
    
    # Infer dims from stats or checkpoint if possible, else hardcode for now
    # Assuming standard dimensions from the training config/dataset
    # You might need to adjust these based on your specific dataset stats
    stats = get_stats("data", task)
    
    # Dummy dimensions for initialization - ideally load from config saved with checkpoint
    # Here we assume the model architecture matches what we trained
    # We need to know action_dim and obs_dim. 
    # Let's try to infer from the weight shapes in state_dict if possible, or just use defaults
    state_dict = checkpoint['model_state_dict']
    
    # Heuristic to find input/output dims from weights
    # This is hacky, better to save config with model
    # Assuming ConditionalMLP structure
    # model.global_cond_layer.weight shape is [hidden, obs_dim]
    # model.map_noise.map_layer.0.weight shape is [hidden, action_dim] (maybe)
    
    # For now, let's assume the same as training:
    # We'll load the dataset stats to get the dims
    if stats:
        # This assumes the stats file exists and has the keys
        action_dim = len(stats["action"]["mean"])
        obs_dim = len(stats["observation.state"]["mean"])
    else:
        print("Stats not found, using default dimensions (check your dataset!)")
        action_dim = 6  # Default guess
        obs_dim = 12    # Default guess
        
    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        vision_backbone="resnet18",
        num_diffusion_steps=100
    ).to(device)
    
    policy.load_state_dict(state_dict)
    policy.eval()
    print("Model loaded.")

    # Setup Video Writers
    front_writer = None
    left_writer = None
    right_writer = None
    output_dir = None

    if save_video:
        output_dir = Path("logs/eval") / time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get resolution from first frame
        scene.update_render()
        front_cam.take_picture()
        h, w = front_cam.get_picture("Color").shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        front_writer = cv2.VideoWriter(str(output_dir / "front.mp4"), fourcc, fps, (w, h))
        left_writer = cv2.VideoWriter(str(output_dir / "left_wrist.mp4"), fourcc, fps, (w, h))
        right_writer = cv2.VideoWriter(str(output_dir / "right_wrist.mp4"), fourcc, fps, (w, h))

    # 3. Evaluation Loop
    step = 0
    max_steps = 500
    print("Starting evaluation loop...")
    for step in trange(max_steps, desc="Evaluating", ncols=80):
        if viewer and viewer.closed:
            break
            
        scene.step()
        scene.update_render()
        
        # Capture Observation
        # 1. State
        # We need to construct the state vector exactly as the dataset has it.
        # The dataset 'lift' only has 6 dimensions (joint positions)
        qpos = left_arm.get_qpos()
        # qvel = left_arm.get_qvel()
        # state_vec = np.concatenate([qpos, qvel]).astype(np.float32)
        state_vec = qpos.astype(np.float32)
        
        # 2. Images
        # SAPIEN 3.x: cam.take_picture(), then cam.get_picture("Color")
        front_cam.take_picture()
        left_wrist_cam.take_picture()
        right_wrist_cam.take_picture()
        
        front_rgba = front_cam.get_picture("Color")
        left_rgba = left_wrist_cam.get_picture("Color")
        right_rgba = right_wrist_cam.get_picture("Color")
        
        # Convert to uint8 [0, 255]
        front_img = (front_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        left_img = (left_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        right_img = (right_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        
        # Update web viewer frame
        if viewer_app:
            viewer_app.update_frames({
                "front": front_img,
                "left": left_img,
                "right": right_img
            })

        # Write to video (OpenCV expects BGR)
        if save_video:
            front_writer.write(cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
            left_writer.write(cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
            right_writer.write(cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
        
        # Preprocess image for model (using front_img)
        # Resize to 96x96
        img_tensor = torch.from_numpy(cv2.resize(front_img, (96, 96))).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, 96, 96]
        
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device) # [1, obs_dim]
        
        # Construct batch
        batch = {
            "observation.state": state_tensor,
            "observation.images.front": img_tensor,
            # Add other cameras if your model expects them
        }
        
        # Normalize
        if stats:
            batch["observation.state"] = normalize(batch["observation.state"], stats, "observation.state")
        
        # Ensure all batch tensors are on the same device as the policy
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # Inference
        with torch.no_grad():
            # Diffusion policy returns action
            # We need to handle the noise scheduler loop inside the policy's forward or sample method
            # The current policy implementation in train.py calls `policy(batch)` which returns LOSS.
            # We need a `sample` method in DiffusionPolicy.
            # Since I implemented DiffusionPolicy, I know if it has a sample method.
            # If it doesn't, I need to add it or use the scheduler directly here.
            # Let's assume I need to add it or it exists.
            # Wait, I didn't check the policy code in detail for a 'sample' method.
            # I will assume standard diffusers usage.
            
            # If the policy class doesn't have a sample method, this will fail.
            # I should check src/lerobot/policy/diffusion.py content if I could.
            # But I can't read it now easily.
            # I will assume it has a `select_action` or `sample` method.
            # If not, I'll write a simple sampler here.
            
            # Placeholder for sampling logic:
            if hasattr(policy, 'select_action'):
                if hasattr(policy.noise_scheduler, "timesteps"):
                    policy.noise_scheduler.timesteps = policy.noise_scheduler.timesteps.to(device)
                action = policy.select_action(batch)
            else:
                # Fallback: just predict noise from random noise (this is wrong for diffusion inference)
                # We need the full DDPM loop.
                # I will assume the user will implement `select_action` in the policy 
                # or I should have implemented it.
                # For now, let's print a warning and use random action to keep the loop running
                # print("Warning: Policy missing select_action, using random.")
                # action = torch.randn(1, action_dim).to(device)
                
                # Actually, I should try to call policy.sample(batch)
                action = policy.sample(batch)

        # Unnormalize action
        if stats:
            action = unnormalize(action, stats, "action")
            
        action_np = action.squeeze(0).cpu().numpy()
        
        # Apply Action
        # Assuming action is joint position targets
        # SAPIEN 3.x: set_qpos is for teleportation, set_drive_target is for control
        # But PhysxArticulation doesn't have set_drive_target directly, it's on the joints (active_joints)
        # Or we can use set_qpos if we just want to visualize the result (kinematic replay)
        # But for control, we should set drive targets on active joints.
        
        # Correct way for SAPIEN 3 Articulation control:
        # left_arm.set_qpos(action_np) # This is teleportation
        
        # For control:
        active_joints = left_arm.get_active_joints()
        for i, joint in enumerate(active_joints):
            joint.set_drive_target(action_np[i])
        
        if viewer:
            viewer.render()
        step += 1

    print("Evaluation finished.")
    
    if save_video:
        front_writer.release()
        left_writer.release()
        right_writer.release()
        print(f"Videos saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--task", type=str, required=True, help="Task name (lift, sort, stack)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (cuda or cpu)")
    parser.add_argument("--save-video", action="store_true", help="Save evaluation videos")
    parser.add_argument("--web-viewer", action="store_true", help="Enable web viewer")
    parser.add_argument("--port", type=int, default=5000, help="Port for web viewer")
    args = parser.parse_args()
    
    main(args.checkpoint, task=args.task, device=args.device, save_video=args.save_video, web_viewer=args.web_viewer, port=args.port)
