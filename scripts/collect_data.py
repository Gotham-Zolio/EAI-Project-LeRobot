import tyro
import numpy as np
import os
import h5py
from lerobot.envs.gym_env import LeRobotGymEnv
from lerobot.policy.scripted import LiftPolicy

def collect_data(
    task: str = "Lift",
    num_episodes: int = 10,
    save_dir: str = "data/raw",
    headless: bool = True
):
    print(f"Collecting {num_episodes} episodes for task {task}...")
    env = LeRobotGymEnv(task=task, headless=headless)
    
    # Initialize Expert Policy
    policy = LiftPolicy(env)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task}_demo.h5")
    
    # Placeholder for data storage
    all_obs = []
    all_actions = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_obs = []
        ep_actions = []
        
        # Reset policy state for new episode
        policy.stage = "APPROACH"
        policy.gripper_state = 1.0
        
        print(f"Episode {ep+1}/{num_episodes}")
        
        while not done:
            # Get action from Expert Policy
            action = policy.get_action(obs)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_obs.append(obs['qpos'])
            ep_actions.append(action)
            obs = next_obs
            
        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_actions))

    env.close()
    
    # Save to HDF5 (Simple format)
    with h5py.File(save_path, 'w') as f:
        # Create groups for each episode or stack them
        # Here we just save as a flat list of episodes for simplicity
        for i, (o, a) in enumerate(zip(all_obs, all_actions)):
            grp = f.create_group(f"episode_{i}")
            grp.create_dataset("qpos", data=o)
            grp.create_dataset("action", data=a)
            
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    tyro.cli(collect_data)
