#!/usr/bin/env python3
"""
Expert demonstration data collector using FSM (finite state machine) and IK (inverse kinematics) expert policy.

Core principles:
1. FSM: Defines discrete task stages and transitions.
2. IK: Converts end-effector pose targets to joint angles.
3. Expert policy: Implements deterministic behavior via hard-coded state machine.

Features:
- Pure FSM implementation, no RL residuals required.
- Concise code structure, focused on core logic.
- Supports lift, stack, and sort tasks.
"""

import sys
import os
import tyro
import json
from datetime import datetime
import numpy as np
import h5py
from dataclasses import dataclass

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.web_viewer.viewer import WebViewer

from lerobot.envs.gym_env import LeRobotGymEnv
from lerobot.policy.task_planner import solve_lift, solve_stack, solve_sort


@dataclass
class CollectionConfig:
    """
    Data collection configuration.
    """
    task: str = "lift"  # Task type: lift, stack, sort
    num_episodes: int = 50  # Number of episodes to collect (append to existing)
    version: str = "v1.0"  # Version string for dataset tracking
    save_dir: str = str(Path(__file__).resolve().parents[1] / "data" / "datasets")  # data/datasets/{task}
    max_steps: int = 300  # Max steps per episode
    headless: bool = True  # Headless mode
    verbose: bool = False  # Verbose output
    web_viewer: bool = False  # Enable web visualization
    port: int = 5000  # Web viewer port
    sleep_viewer_sec: float = 0.1  # Viewer refresh interval
    vis: bool = False  # Visualize axes in simulation



from typing import List, Dict, Any, Tuple, Optional

class RecordingWrapper:
    """
    Environment wrapper for automatic data recording during step.
    Records qpos, action, reward, done, and images for each step.
    """
    def __init__(self, env: Any, cameras: List[str], viewer_app: Optional[Any] = None, sleep_viewer_sec: float = 0.1):
        self.env = env
        self.cameras = cameras
        self.viewer_app = viewer_app
        self.sleep_viewer_sec = sleep_viewer_sec
        self.trajectory: Dict[str, Any] = {
            "qpos": [], "action": [], "reward": [], "done": [],
            "images": {cam: [] for cam in cameras}
        }
        self.step_count = 0
        self.last_obs: Optional[Dict[str, Any]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.last_obs = obs
        # Clear trajectory
        self.trajectory = {
            "qpos": [], "action": [], "reward": [], "done": [],
            "images": {cam: [] for cam in self.cameras}
        }
        self.step_count = 0
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.last_obs is None:
            # Should have been set by reset, but if not (e.g. manual reset of inner env), try to get it
            self.last_obs = self.env._get_obs()

        obs = self.last_obs

        # WebViewer update
        if self.viewer_app:
            frames = {cam: obs["images"][cam] for cam in self.cameras if cam in obs["images"]}
            # Optionally push world view to WebViewer
            if "world" in obs["images"]:
                frames["world"] = obs["images"]["world"]
            self.viewer_app.update_frames(frames)
            # Optionally slow down for viewer
            # import time; time.sleep(self.sleep_viewer_sec)

        # Take action
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Record data
        self.trajectory["qpos"].append(obs["qpos"])
        self.trajectory["action"].append(action)
        self.trajectory["reward"].append(reward)
        self.trajectory["done"].append(done)
        for cam in self.cameras:
            if cam in obs["images"]:
                self.trajectory["images"][cam].append(obs["images"][cam])

        self.last_obs = next_obs
        self.step_count += 1

        return next_obs, reward, terminated, truncated, info


class FSMDataCollector:
    """
    Data collector using motion planning (FSM+IK expert policy).
    """
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.env = None
        self.solver = None
        self.viewer_app = None
        self.cameras: List[str] = []

    def setup(self) -> None:
        """
        Initialize environment and solver.
        """
        print(f"Initializing task: {self.config.task}")

        # Camera configuration
        if self.config.task == "lift":
            self.cameras = ["front", "right_wrist"]
        elif self.config.task == "sort":
            self.cameras = ["front", "left_wrist", "right_wrist"]
        elif self.config.task == "stack":
            self.cameras = ["front", "right_wrist"]
        else:
            raise ValueError(f"Unknown task: {self.config.task}")

        print(f"Using cameras: {self.cameras}")

        # Start WebViewer if needed
        if self.config.web_viewer:
            self.viewer_app = WebViewer(port=self.config.port)
            self.viewer_app.start()
            print(f"Web viewer started at http://localhost:{self.config.port}")

        # Create environment
        self.env = LeRobotGymEnv(
            task=self.config.task,
            headless=self.config.headless,
            max_steps=self.config.max_steps
        )

        # Select solver
        solver_map = {
            "lift": solve_lift,
            "stack": solve_stack,
            "sort": solve_sort
        }
        self.solver = solver_map[self.config.task]

    def collect_episode(self, episode_id: int) -> Tuple[Dict[str, Any], bool]:
        """
        Collect a single episode.
        """
        wrapper = RecordingWrapper(
            self.env,
            self.cameras,
            self.viewer_app,
            self.config.sleep_viewer_sec
        )

        # Reset environment with episode_id as seed
        wrapper.reset(seed=episode_id)

        print(f"\nEpisode {episode_id + 1}/{self.config.num_episodes}")
        if self.viewer_app:
            self.viewer_app.update_status(
                mode="Data Collection (MP)",
                episode=episode_id + 1,
                total_episodes=self.config.num_episodes,
                task=self.config.task,
            )

        # Run solver
        try:
            self.solver(wrapper, seed=episode_id, debug=self.config.verbose, vis=self.config.vis)
            success = True
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

        # Get trajectory
        trajectory = wrapper.trajectory

        # Append final observation images
        if wrapper.last_obs is not None:
            for cam in self.cameras:
                if cam in wrapper.last_obs["images"]:
                    trajectory["images"][cam].append(wrapper.last_obs["images"][cam])

        print(f"Episode finished: steps={len(trajectory['action'])}, success={success}")

        return trajectory, success

    def save_data(self, trajectories: List[Dict[str, Any]], save_path: Path) -> None:
        """
        Save trajectory data to HDF5 file.
        """
        print(f"\nSaving data to: {save_path}")
        try:
            mode = "a" if save_path.exists() else "w"
            with h5py.File(save_path, mode) as f:
                # Find next episode index
                start_ep_id = 0
                if "episode_0" in f:
                    existing_eps = [k for k in f.keys() if k.startswith("episode_")]
                    if existing_eps:
                        max_id = max(int(k.split("_")[1]) for k in existing_eps)
                        start_ep_id = max_id + 1

                # Metadata
                total_episodes = start_ep_id + len(trajectories)
                f.attrs["task"] = self.config.task
                f.attrs["version"] = self.config.version
                f.attrs["num_episodes"] = total_episodes
                f.attrs["collection_method"] = "fsm_ik"
                f.attrs["last_updated"] = datetime.now().isoformat()
                if "cameras" not in f.attrs:
                    f.attrs["cameras"] = self.cameras

                # Save each episode
                for i, traj in enumerate(trajectories):
                    ep_id = start_ep_id + i
                    grp = f.create_group(f"episode_{ep_id}")
                    grp.create_dataset("qpos", data=np.array(traj["qpos"], dtype=np.float32))
                    grp.create_dataset("action", data=np.array(traj["action"], dtype=np.float32))
                    grp.create_dataset("reward", data=np.array(traj["reward"], dtype=np.float32))
                    grp.create_dataset("done", data=np.array(traj["done"], dtype=bool))

                    img_grp = grp.create_group("images")
                    for cam in self.cameras:
                        if cam in traj["images"]:
                            img_grp.create_dataset(
                                cam,
                                data=np.array(traj["images"][cam], dtype=np.uint8)
                            )
            print(f"✅ Data saved successfully (total episodes: {total_episodes})")
        except Exception as e:
            print(f"Failed to save data: {e}")

    def run(self) -> None:
        """
        Run the full data collection process.
        """
        self.setup()

        # Prepare save paths following the standard structure
        save_dir = Path(self.config.save_dir) / self.config.task / "raw"
        save_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = Path(self.config.save_dir) / self.config.task / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename: {task}_{version}_{date}.h5
        date_str = datetime.now().strftime("%Y%m%d")
        h5_filename = f"{self.config.task}_{self.config.version}_{date_str}.h5"
        save_path = save_dir / h5_filename

        # Collect episodes (only successful ones)
        trajectories: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0
        episode_id = 0
        
        print(f"\n{'='*60}")
        print(f"Target: {self.config.num_episodes} successful episodes")
        print(f"{'='*60}\n")

        # Keep collecting until we have enough successful episodes
        while success_count < self.config.num_episodes:
            trajectory, success = self.collect_episode(episode_id)
            
            if success:
                trajectories.append(trajectory)
                success_count += 1
                print(f"✅ Episode {episode_id} succeeded ({success_count}/{self.config.num_episodes} collected)")
            else:
                failed_count += 1
                print(f"❌ Episode {episode_id} failed (IK/planning error) - skipping")
            
            episode_id += 1
            
            # Safety limit: prevent infinite loop if too many failures
            if episode_id > self.config.num_episodes * 5:
                print(f"\n⚠️  Warning: Too many failures ({failed_count}). Stopping collection.")
                print(f"Collected {success_count}/{self.config.num_episodes} successful episodes.")
                break

        # Save data (only successful episodes)
        if trajectories:
            self.save_data(trajectories, save_path)
        else:
            print("\n⚠️  No successful episodes to save!")
            return

        # Save metadata
        total_attempts = success_count + failed_count
        success_rate = 100 * success_count / total_attempts if total_attempts > 0 else 0
        total_steps = sum(len(traj["action"]) for traj in trajectories)
        
        info_file = meta_dir / f"{self.config.task}_{self.config.version}_info.json"
        info = {
            "task": self.config.task,
            "version": self.config.version,
            "created_date": datetime.now().isoformat(),
            "episodes_collected": success_count,
            "episodes_attempted": total_attempts,
            "episodes_failed": failed_count,
            "total_steps": total_steps,
            "success_rate": success_rate,
            "cameras": self.cameras,
            "fps": 30,
            "h5_file": h5_filename,
        }
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
        print(f"✅ Metadata saved to {info_file}")

        # Summary
        print(f"\n{'='*60}")
        print(f"Collection finished")
        print(f"Task: {self.config.task} (v{self.config.version})")
        print(f"Successfully collected: {success_count} episodes")
        print(f"Total attempts: {total_attempts} (failed: {failed_count})")
        print(f"Total steps: {total_steps}")
        print(f"Success rate: {success_count}/{total_attempts} ({success_rate:.1f}%)")
        print(f"Data saved to: {save_path}")
        print(f"Metadata saved to: {info_file}")
        print(f"💡 Tip: Run again with same task/version to append more episodes")
        print(f"        Or increment version (e.g., v1.1) for a new batch")
        print(f"{'='*60}")

        # Close environment
        self.env.close()
        if self.viewer_app:
            print("Closing WebViewer...")
            # No stop method, thread daemon auto-exits



def _adapt_hydra_style_args(argv: list[str]) -> list[str]:
    """Allow Hydra-style key=value by converting to --config.key value for Tyro."""
    adapted: list[str] = []
    for arg in argv:
        if "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            adapted.extend([f"--config.{key}", value])
        else:
            adapted.append(arg)
    return adapted


def main(config: CollectionConfig):
    """
    Entry point for FSM+IK expert data collection.
    """
    collector = FSMDataCollector(config)
    collector.run()


if __name__ == "__main__":
    parsed_args = _adapt_hydra_style_args(sys.argv[1:])
    tyro.cli(main, args=parsed_args)
