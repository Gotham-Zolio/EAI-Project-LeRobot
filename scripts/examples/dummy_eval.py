import gymnasium as gym
from typing import Dict, Any, Literal
import numpy as np
import dataclasses
import tyro

# single arm observation space
# images {'front': Box(0, 255, (480, 640, 3), uint8), 'wrist': Box(0, 255, (480, 640, 3), uint8)}
# states {'left_arm': Box(-np.inf, np.inf, (6,), float32), 'right_arm': Box(-np.inf, np.inf, (6,), float32)} --- IGNORE ---
# actions Box(-np.inf, np.inf, (6,), float32)

# dual arm observation space
# images {'front': Box(0, 255, (480, 640, 3), uint8), 'left_wrist': Box(0, 255, (480, 640, 3), uint8), 'right_wrist': Box(0, 255, (480, 640, 3), uint8)}
# states {'arm': Box(-np.inf, np.inf, (6,), float32)}
# actions Box(-np.inf, np.inf, (12,), float32) [left, right]
class DummyEnv(gym.Env):
    def __init__(self, mode: Literal['single_arm', 'dual_arm'] = 'single_arm'):
        super(DummyEnv, self).__init__()
        if mode == 'single_arm':
            self.observation_space = gym.spaces.Dict({
                'images': gym.spaces.Dict({
                    'front': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                    'wrist': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                }),
                "state": gym.spaces.Dict({
                    'left_arm': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                    'right_arm': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                }),
            })
            self.action_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        elif mode == 'dual_arm':
            self.observation_space = gym.spaces.Dict({
                'images': gym.spaces.Dict({
                    'front': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                    'left_wrist': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                    'right_wrist': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                }),
                "state": gym.spaces.Dict({
                    'arm': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                }),
            })
            self.action_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        else:
            raise ValueError("mode must be 'single_arm' or 'dual_arm'")

    def get_observation(self) -> Dict[str, Any]:
        return self.observation_space.sample()
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        observation = self.get_observation()
        info = {}
        return observation, info
    
    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation = self.get_observation()
        reward = np.random.rand()
        terminated = np.random.rand() < 0.05  # 5% chance to terminate each step
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
    
@dataclasses.dataclass
class DummyPolicyConfig:
    policy_type: Literal['single_arm', 'dual_arm'] = 'single_arm'
    action_horizon: int = 16
    
class DummyPolicy:
    def __init__(self, config: DummyPolicyConfig):
        if config.policy_type == 'single_arm':
            self.action_dim = 6
        elif config.policy_type == 'dual_arm':
            self.action_dim = 12
        else:
            raise ValueError("policy_type must be 'single_arm' or 'dual_arm'")
        self.action_horizon = config.action_horizon
        
    def reset(self, **kwargs):
        print("Policy reset called.")
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(self.action_horizon, self.action_dim)).astype(np.float32)

def main(num_episodes: int, dummy_policy: DummyPolicyConfig, prompt: str = 'hello EAI!', env_mode: Literal['single_arm', 'dual_arm'] = 'single_arm'):
    env = DummyEnv(mode=env_mode)
    policy = DummyPolicy(config=dummy_policy)

    for episode in range(num_episodes):
        done = False
        step_count = 0
        obs, info = env.reset()
        policy.reset()
        while not done:
            obs['prompt'] = prompt
            action_seq = policy.get_action(obs)
            for action in action_seq:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                if done:
                    break
        print(f"Episode {episode + 1} finished in {step_count} steps.")
    
if __name__ == "__main__":
    tyro.cli(main)