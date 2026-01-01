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
    num_episodes: int = 10  # Number of episodes to collect
    save_dir: str = str(Path(__file__).resolve().parents[1] / "data")  # Save directory under repo/data
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
            with h5py.File(save_path, "w") as f:
                # Metadata
                f.attrs["task"] = self.config.task
                f.attrs["num_episodes"] = len(trajectories)
                f.attrs["collection_method"] = "fsm_ik"
                f.attrs["cameras"] = self.cameras

                # Save each episode
                for ep_id, traj in enumerate(trajectories):
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
            print(f"✅ Data saved successfully")
        except Exception as e:
            print(f"Failed to save data: {e}")

    def run(self) -> None:
        """
        Run the full data collection process.
        """
        self.setup()

        # Prepare save path
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{self.config.task}_demo.h5"

        # Collect all episodes
        trajectories: List[Dict[str, Any]] = []
        success_count = 0

        for ep in range(self.config.num_episodes):
            trajectory, success = self.collect_episode(ep)
            trajectories.append(trajectory)
            if success:
                success_count += 1

        # Save data
        self.save_data(trajectories, save_path)

        # Summary
        print(f"\n{'='*60}")
        print(f"Collection finished")
        print(f"Task: {self.config.task}")
        print(f"Total episodes: {self.config.num_episodes}")
        print(f"Success rate: {success_count}/{self.config.num_episodes} ({100*success_count/self.config.num_episodes:.1f}%)")
        print(f"Save path: {save_path}")
        print(f"{'='*60}")

        # Close environment
        self.env.close()
        if self.viewer_app:
            print("Closing WebViewer...")
            # No stop method, thread daemon auto-exits



def main(config: CollectionConfig):
    """
    Entry point for FSM+IK expert data collection.
    """
    collector = FSMDataCollector(config)
    collector.run()


if __name__ == "__main__":
    tyro.cli(main)
