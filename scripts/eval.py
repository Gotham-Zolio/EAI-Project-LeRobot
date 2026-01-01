import sys
import os
import time
import argparse
import numpy as np
import torch
from pathlib import Path
import json

# Add src and tools to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
tools_path = os.path.join(current_dir, "..")
sys.path.insert(0, src_path)
sys.path.insert(0, tools_path)

from lerobot.envs.gym_env import LeRobotGymEnv
from lerobot.policy.diffusion import DiffusionPolicy
from tools.web_viewer.viewer import WebViewer


def load_stats_from_checkpoint_dir(checkpoint_path):
    """Load stats.json from the same directory as the checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent
    stats_path = checkpoint_dir / "stats.json"
    
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print(f"Loaded stats from {stats_path}")
        return stats
    else:
        print(f"Warning: stats.json not found at {stats_path}")
        return None


def normalize(data, stats, key):
    if stats is None or key not in stats:
        return data
    mean = torch.tensor(stats[key]["mean"], device=data.device, dtype=data.dtype)
    std = torch.tensor(stats[key]["std"], device=data.device, dtype=data.dtype)
    return (data - mean) / (std + 1e-8)


def unnormalize(data, stats, key):
    if stats is None or key not in stats:
        return data
    mean = torch.tensor(stats[key]["mean"], device=data.device, dtype=data.dtype)
    std = torch.tensor(stats[key]["std"], device=data.device, dtype=data.dtype)
    return data * (std + 1e-8) + mean



def main(checkpoint_path, task, num_episodes=10, device="cuda", headless=False, web_viewer=True, port=5000):
    """
    Evaluate a trained DiffusionPolicy on the specified task.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        task: Task name (lift, sort, stack)
        num_episodes: Number of episodes to evaluate
        device: Device to run on (cuda or cpu)
        headless: Run without GUI viewer
        web_viewer: Enable web viewer for visualization
        port: Port for web viewer
    """
    # Start web viewer if requested
    viewer_app = None
    if web_viewer:
        viewer_app = WebViewer(port=port)
        viewer_app.start()
        print(f"Web viewer started at http://localhost:{port}")

    # Load stats from checkpoint directory
    stats = load_stats_from_checkpoint_dir(checkpoint_path)
    
    if stats is None:
        print("ERROR: Cannot proceed without stats.json. Make sure it was saved during training.")
        return

    # Infer dimensions from stats
    action_dim = len(stats["action"]["mean"])
    obs_dim = len(stats["observation.state"]["mean"])
    print(f"Model dimensions: action_dim={action_dim}, obs_dim={obs_dim}")

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
        vision_backbone="resnet18",
        num_diffusion_steps=100
    ).to(device)
    
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    print("✅ Model loaded successfully")

    # Create environment
    print(f"Creating environment for task: {task}")
    env = LeRobotGymEnv(
        task=task,
        headless=headless,
        max_steps=1000  # 增加到1000步
    )

    # Evaluation metrics
    success_count = 0
    episode_lengths = []
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation: {num_episodes} episodes")
    print(f"{'='*60}\n")

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        truncated = False
        step_count = 0
        
        print(f"Episode {ep + 1}/{num_episodes}")
        
        while not (done or truncated):
            # Prepare batch for policy
            state_tensor = torch.from_numpy(obs["qpos"]).float().unsqueeze(0).to(device)
            
            # Get front camera image
            front_img = obs["images"]["front"]  # RGB or RGBA uint8
            
            # Handle RGBA images (4 channels) -> RGB (3 channels)
            if front_img.shape[-1] == 4:
                front_img = front_img[..., :3]
            
            # Policy expects (1, 3, H, W) in [0, 1] range
            img_tensor = torch.from_numpy(front_img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            batch = {
                "observation.state": state_tensor,
                "observation.images.front": img_tensor,
            }
            
            # Normalize
            batch["observation.state"] = normalize(batch["observation.state"], stats, "observation.state")
            
            # Update web viewer
            if viewer_app:
                frames = {"front": front_img}
                # Add all available camera views
                for cam_name in ["right_wrist", "left_wrist", "world"]:
                    if cam_name in obs["images"]:
                        cam_img = obs["images"][cam_name]
                        # Handle RGBA -> RGB
                        if cam_img.shape[-1] == 4:
                            cam_img = cam_img[..., :3]
                        frames[cam_name] = cam_img
                viewer_app.update_frames(frames)
                viewer_app.update_status(
                    mode=f"Evaluation (Policy) Step {step_count}",
                    episode=ep + 1,
                    total_episodes=num_episodes,
                    task=task
                )
            
            # Inference
            with torch.no_grad():
                action = policy.select_action(batch)
            
            # Unnormalize action
            action = unnormalize(action, stats, "action")
            action_np = action.squeeze(0).cpu().numpy()
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action_np)
            step_count += 1
            
        
        # Episode finished
        success = info.get("success", False)
        episode_lengths.append(step_count)
        
        if success:
            success_count += 1
            print(f"  ✅ Success after {step_count} steps")
        else:
            print(f"  ❌ Failed after {step_count} steps")
    
    # Final statistics
    success_rate = 100 * success_count / num_episodes
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_rate:.1f}%)")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Save results
    results_dir = Path(checkpoint_path).parent / "eval_results"
    results_dir.mkdir(exist_ok=True)
    
    results = {
        "task": task,
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_rate,
        "average_episode_length": avg_length,
        "episode_lengths": episode_lengths,
        "checkpoint": str(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_file = results_dir / f"eval_{task}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {results_file}")
    
    env.close()
    if viewer_app:
        print("Web viewer will remain open. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DiffusionPolicy")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to .pth checkpoint (e.g., logs/train/lift/.../checkpoint_epoch_100.pth)")
    parser.add_argument("--task", type=str, required=True, 
                       help="Task name (lift, sort, stack)")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to run on (cuda or cpu, default: cuda)")
    parser.add_argument("--headless", action="store_true", 
                       help="Run without GUI viewer")
    parser.add_argument("--no-web-viewer", dest='web_viewer', action="store_false", default=True,
                       help="Disable web viewer")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port for web viewer (default: 5000)")
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint,
        task=args.task,
        num_episodes=args.num_episodes,
        device=args.device,
        headless=args.headless,
        web_viewer=args.web_viewer,
        port=args.port
    )
