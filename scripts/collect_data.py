import sys
import os
import time

# =====================
# Add src and tools to path
# =====================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../src")
tools_path = os.path.join(current_dir, "..")
sys.path.insert(0, src_path)
sys.path.insert(0, tools_path)

import tyro
import numpy as np
import h5py

from lerobot.envs.gym_env import LeRobotGymEnv
from lerobot.policy.scripted import LiftPolicy, SortPolicy, StackPolicy
from tools.web_viewer.viewer import WebViewer

# Optional: stable-baselines3 for loading RL models
try:
    from stable_baselines3 import PPO, SAC, TD3
except Exception:
    PPO = SAC = TD3 = None

# =====================
# Constants
# =====================
MAX_STEPS = 300   # 防止策略异常导致死循环


class PhaseResiduals:
    """Loader and predictor for per-phase residual RL models.

    Expects model files under: models/phased_rl/<task>/phase_<id>.zip
    Supports PPO/SAC/TD3 if stable-baselines3 is available.
    """
    def __init__(self, task: str, models_root: str, enable_mask: tuple):
        from pathlib import Path
        self.task = task
        self.models_root = Path(models_root) / task
        self.enable_mask = enable_mask
        self.models = {}
        self._load_models()

    def _load_models(self):
        if PPO is None and SAC is None and TD3 is None:
            print("[PhaseResiduals] stable-baselines3 not available; residuals disabled.")
            return
        for phase_id in range(0, 6):
            if phase_id < len(self.enable_mask) and not self.enable_mask[phase_id]:
                continue
            for algo_name, loader in (("ppo", PPO), ("sac", SAC), ("td3", TD3)):
                if loader is None:
                    continue
                model_path = self.models_root / f"phase_{phase_id}.zip"
                if model_path.exists():
                    try:
                        self.models[phase_id] = loader.load(str(model_path))
                        print(f"[PhaseResiduals] Loaded {algo_name.upper()} model for phase {phase_id}: {model_path}")
                        break
                    except Exception as e:
                        print(f"[PhaseResiduals] Failed to load {model_path}: {e}")
        if not self.models:
            print("[PhaseResiduals] No per-phase models found; running base scripted policy only.")

    def has(self, phase_id: int) -> bool:
        return phase_id in self.models

    def residual(self, phase_id: int, features: np.ndarray) -> np.ndarray | None:
        if phase_id not in self.models:
            return None
        try:
            action, _ = self.models[phase_id].predict(features, deterministic=True)
            return np.asarray(action, dtype=np.float32)
        except Exception as e:
            print(f"[PhaseResiduals] Predict failed for phase {phase_id}: {e}")
            return None


def build_features(obs: dict, phase_id: int, policy) -> np.ndarray:
    """Build residual policy features from observation and current phase.

    Features: [qpos, phase_onehot(6), last_target_pos(3)]
    """
    qpos = np.asarray(obs.get("qpos", []), dtype=np.float32)
    phase_onehot = np.zeros(6, dtype=np.float32)
    if 0 <= phase_id < 6:
        phase_onehot[phase_id] = 1.0
    target_pos = getattr(policy, "last_target_pos", np.zeros(3, dtype=np.float32))
    return np.concatenate([qpos, phase_onehot, target_pos], dtype=np.float32)

# =====================
# Main Data Collection
# =====================
def collect_data(
    task: str = "lift",
    num_episodes: int = 10,
    save_dir: str = "data/raw",
    headless: bool = True,
    web_viewer: bool = False,
    port: int = 5000,
    policy_verbose: bool = False,
    flip_wrist_flex: bool = False,
    # Residual RL options
    residual_scale: float = 0.2,
    enable_phases: tuple = (True, True, True, True, True, True),
    models_root: str = "models/phased_rl",
    sleep_viewer_sec: float = 0.1,
):
    task = task.lower()
    print(f"Collecting {num_episodes} episodes for task '{task}'")

    # ---------- Web Viewer ----------
    viewer_app = None
    if web_viewer:
        viewer_app = WebViewer(port=port)
        viewer_app.start()
        print(f"Web viewer started at http://localhost:{port}")

    # ---------- Environment ----------
    env = LeRobotGymEnv(task=task, headless=headless)

    # ---------- Policy & camera config ----------
    if task == "lift":
        policy_cls = LiftPolicy
        desired_cameras = ["front", "right_wrist"]
    elif task == "sort":
        policy_cls = SortPolicy
        desired_cameras = ["front", "left_wrist", "right_wrist"]
    elif task == "stack":
        policy_cls = StackPolicy
        desired_cameras = ["front", "right_wrist"]
    else:
        raise ValueError(f"Unknown task: {task}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task}_demo.h5")

    # =====================
    # HDF5 File
    # =====================
    # Residuals loader (optional)
    residuals = PhaseResiduals(task, models_root, enable_phases)

    with h5py.File(save_path, "w") as f:
        f.attrs["task"] = task
        f.attrs["collection_method"] = "phased_rl"
        f.attrs["residual_scale"] = residual_scale

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step_count = 0

            # ---------- Resolve actual camera keys ----------
            available_cameras = list(obs["images"].keys())
            cameras = [c for c in desired_cameras if c in available_cameras]

            if ep == 0:
                print("Available cameras:", available_cameras)
                print("Using cameras:", cameras)
                f.attrs["cameras"] = cameras

            # ---------- Buffers ----------
            ep_qpos = []
            ep_qpos_next = []
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            ep_images = {cam: [] for cam in cameras}

            # ---------- Reset policy ----------
            policy = policy_cls(env)
            
            # Enable verbose mode to see IK debug output (helpful for troubleshooting)
            # Set to True to see joint angles and target positions
            policy.verbose = policy_verbose
            # Note: wrist_flex flipping is handled internally in LiftPolicy during XY_ALIGN

            print(f"\nEpisode {ep + 1}/{num_episodes}")

            if viewer_app:
                viewer_app.update_status(
                    mode="Data Collection",
                    episode=ep + 1,
                    total_episodes=num_episodes,
                    task=task,
                )

            # =====================
            # Rollout
            # =====================
            while not done and step_count < MAX_STEPS:
                step_count += 1

                # ---- Viewer ----
                if viewer_app:
                    frames = {cam: obs["images"][cam] for cam in cameras}
                    # Debug: print frame info on first step
                    if step_count == 1:
                        print(f"  Sending frames to viewer: {list(frames.keys())}")
                        for cam, frame in frames.items():
                            print(f"    {cam}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
                    viewer_app.update_frames(frames)
                    time.sleep(sleep_viewer_sec)

                # ---- Policy ----
                base_action = policy.get_action(obs)
                phase_id = getattr(policy, "phase", -1)

                # Residual RL: build features and add scaled residual if model exists
                residual_action = None
                if residuals.has(phase_id):
                    features = build_features(obs, phase_id, policy)
                    residual_action = residuals.residual(phase_id, features)

                if residual_action is not None:
                    try:
                        low, high = env.action_space.low, env.action_space.high
                    except Exception:
                        low, high = None, None
                    action = np.asarray(base_action, dtype=np.float32)
                    residual_scaled = np.asarray(residual_action, dtype=np.float32) * residual_scale
                    if len(residual_scaled) != len(action):
                        residual_scaled = np.resize(residual_scaled, len(action))
                    action = action + residual_scaled
                    if low is not None and high is not None:
                        action = np.clip(action, low, high)
                else:
                    action = base_action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # ---- Record transition (s_t, a_t, s_{t+1}) ----
                ep_qpos.append(obs["qpos"])
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_dones.append(done)

                for cam in cameras:
                    ep_images[cam].append(obs["images"][cam])

                obs = next_obs

            # ---------- Store final state s_{T} ----------
            ep_qpos_next = ep_qpos[1:] + [obs["qpos"]]

            for cam in cameras:
                ep_images[cam].append(obs["images"][cam])

            # =====================
            # Save episode
            # =====================
            grp = f.create_group(f"episode_{ep}")

            grp.create_dataset("qpos", data=np.asarray(ep_qpos, dtype=np.float32))
            grp.create_dataset("qpos_next", data=np.asarray(ep_qpos_next, dtype=np.float32))
            grp.create_dataset("action", data=np.asarray(ep_actions, dtype=np.float32))
            grp.create_dataset("reward", data=np.asarray(ep_rewards, dtype=np.float32))
            grp.create_dataset("done", data=np.asarray(ep_dones, dtype=np.bool_))

            img_grp = grp.create_group("images")
            for cam in cameras:
                img_grp.create_dataset(cam, data=np.asarray(ep_images[cam]))

            print(
                f"Episode {ep + 1} finished | "
                f"steps={step_count} | "
                f"success={any(ep_rewards)}"
            )

    env.close()
    print(f"\n✅ Data saved to {save_path}")

# =====================
# Entry
# =====================
if __name__ == "__main__":
    tyro.cli(collect_data)
