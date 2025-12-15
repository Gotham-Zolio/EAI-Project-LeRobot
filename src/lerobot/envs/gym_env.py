import gymnasium as gym
import numpy as np
import sapien
from gymnasium import spaces
from lerobot.envs.sapien_env import create_scene, setup_scene
from lerobot.common.camera import apply_distortion, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY

class LeRobotGymEnv(gym.Env):
    def __init__(self, task="lift", headless=True, max_steps=500):
        super().__init__()
        self.task = task.lower()
        self.headless = headless
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize Scene
        self.scene, self.front_cam, self.left_arm, self.right_arm, \
        self.left_wrist_cam, self.right_wrist_cam = create_scene(headless=headless)
        
        # Setup Task Specifics
        setup_scene(self.scene, self.task)

        # Setup Robot Controllers (Joint Position Control)
        self._setup_controllers(self.left_arm)
        self._setup_controllers(self.right_arm)

        # Define Action Space: 7 joints * 2 arms = 14 dim (normalized -1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        
        # Define Observation Space
        self.observation_space = spaces.Dict({
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            "images": spaces.Dict({
                "front": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "left_wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "right_wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
            })
        })

    def _setup_controllers(self, robot):
        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=1000, damping=200)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Reset robot pose (simplified)
        # Assuming 6 DOF (5 arm + 1 gripper) but we use 7 for compatibility
        # We need to check actual DOF. For now, use zeros of correct length.
        init_qpos_left = np.zeros(self.left_arm.dof)
        init_qpos_right = np.zeros(self.right_arm.dof)
        self.left_arm.set_qpos(init_qpos_left)
        self.right_arm.set_qpos(init_qpos_right)
        
        # Step simulation to settle
        for _ in range(10):
            self.scene.step()
            
        return self._get_obs(), {}

    def step(self, action):
        # Action is target joint positions
        # Split action for left and right arm
        # Assuming action is 14 dims (7+7)
        # But robot might have 6 DOF. We take first N.
        
        left_dof = self.left_arm.dof
        right_dof = self.right_arm.dof
        
        left_action = action[:left_dof]
        right_action = action[7:7+right_dof]
        
        self.left_arm.set_drive_target(left_action)
        self.right_arm.set_drive_target(right_action)
        
        # Step physics
        for _ in range(4): # Frame skip
            self.left_arm.set_qf(self.left_arm.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.right_arm.set_qf(self.right_arm.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.scene.step()
            
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False # Define success condition here
        reward = 0.0 # Define reward function here
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Capture images
        self.scene.update_render()
        
        self.front_cam.take_picture()
        front_img = (self.front_cam.get_picture("Color") * 255).astype(np.uint8)
        # Apply distortion to front camera
        front_img = apply_distortion(front_img, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY)
        
        self.left_wrist_cam.take_picture()
        left_wrist_img = (self.left_wrist_cam.get_picture("Color") * 255).astype(np.uint8)
        
        self.right_wrist_cam.take_picture()
        right_wrist_img = (self.right_wrist_cam.get_picture("Color") * 255).astype(np.uint8)
        
        # Proprioception
        left_qpos = self.left_arm.get_qpos()
        right_qpos = self.right_arm.get_qpos()
        
        # Pad qpos to 14 if needed for consistency
        full_qpos = np.zeros(14, dtype=np.float32)
        full_qpos[:len(left_qpos)] = left_qpos
        full_qpos[7:7+len(right_qpos)] = right_qpos
        
        return {
            "qpos": full_qpos,
            "images": {
                "front": front_img,
                "left_wrist": left_wrist_img,
                "right_wrist": right_wrist_img
            }
        }

    def close(self):
        self.scene = None
