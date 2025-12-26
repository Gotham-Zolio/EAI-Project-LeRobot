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
        self.task_actors = []
        
        # Initialize Scene
        self.scene, self.front_cam, self.left_arm, self.right_arm, self.left_wrist_cam, self.right_wrist_cam, self.world_cam = create_scene(headless=headless)
        
        # Setup Task Specifics
        self.task_actors = setup_scene(self.scene, self.task)

        # Setup Robot Controllers (Joint Position Control)
        self._setup_controllers(self.left_arm)
        self._setup_controllers(self.right_arm)

        # Cache a "home" joint configuration from the scene creation.
        # This is the same ready pose defined inside create_scene/load_arm,
        # which is already natural and suitable for grasping.
        self.init_qpos_left = self.left_arm.get_qpos().copy()
        self.init_qpos_right = self.right_arm.get_qpos().copy()

        # Define Action Space: actual DOF * 2 arms (e.g., 6+6=12 for SO-101)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.left_arm.dof + self.right_arm.dof,), dtype=np.float32)
        
        # Define Observation Space
        self.observation_space = spaces.Dict({
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.left_arm.dof + self.right_arm.dof,), dtype=np.float32),
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
        
        # Remove old task actors
        for actor in self.task_actors:
            self.scene.remove_actor(actor)
        self.task_actors = []
        
        # Re-populate scene (randomized)
        self.task_actors = setup_scene(self.scene, self.task)
        
        # Reset robot pose to the cached "home" configuration instead of zeros,
        # so that each episode starts from a natural grasp-ready posture.
        self.left_arm.set_qpos(self.init_qpos_left)
        self.right_arm.set_qpos(self.init_qpos_right)
        
        # Step simulation to settle
        for _ in range(10):
            self.scene.step()
            
        return self._get_obs(), {}

    def step(self, action):
        # Action is target joint positions
        # Split action for left and right arm based on actual DOF
        
        left_dof = self.left_arm.dof
        right_dof = self.right_arm.dof
        
        left_action = action[:left_dof]
        right_action = action[left_dof:left_dof+right_dof]
        
        for i, joint in enumerate(self.left_arm.get_active_joints()):
            joint.set_drive_target(left_action[i])
        for i, joint in enumerate(self.right_arm.get_active_joints()):
            joint.set_drive_target(right_action[i])
        
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

        self.world_cam.take_picture()
        world_img = (self.world_cam.get_picture("Color") * 255).astype(np.uint8)

        # 调试输出各摄像头画面信息
        # for name, img in zip(["front", "left_wrist", "right_wrist", "world"], [front_img, left_wrist_img, right_wrist_img, world_img]):
        #     try:
        #         print(f"[DEBUG] {name}: shape={img.shape}, min={img.min()}, max={img.max()}, dtype={img.dtype}")  # 屏蔽冗余DEBUG输出，防止刷屏
        #     except Exception as e:
        #         print(f"[DEBUG] {name}: error {e}")

        # Proprioception
        left_qpos = self.left_arm.get_qpos()
        right_qpos = self.right_arm.get_qpos()

        # Concatenate qpos from both arms
        full_qpos = np.concatenate([left_qpos, right_qpos], dtype=np.float32)

        return {
            "qpos": full_qpos,
            "images": {
                "front": front_img,
                "left_wrist": left_wrist_img,
                "right_wrist": right_wrist_img,
                "world": world_img
            }
        }

    def close(self):
        self.scene = None
