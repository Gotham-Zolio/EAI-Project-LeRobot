import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.envs.sapien_env import SO101TaskEnv


class LeRobotGymEnv(gym.Env):
    def __init__(self, task="lift", headless=True, max_steps=500):
        super().__init__()
        self.task = task.lower()
        self.headless = headless
        self.max_steps = max_steps
        self.current_step = 0

        # Create ManiSkill-style environment
        self.env = SO101TaskEnv()
        self.env.reset(seed=None)
        self.scene = self.env.scene
        self.front_cam = None  # Will be handled in _get_obs
        self.world_cam = None  # Will be handled in _get_obs

        # Setup task-specific objects
        self.task_actors = self.env._load_scene({}, task_name=self.task)

        # Cache a "home" joint configuration from the scene creation (defer to after reset)
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        self.init_qpos_left = left_arm.get_qpos().copy()
        self.init_qpos_right = right_arm.get_qpos().copy()

        # Define Action Space: actual DOF * 2 arms (e.g., 6+6=12 for SO-101)
        self.action_space = spaces.Box(low=-1, high=1, shape=(left_arm.dof + right_arm.dof,), dtype=np.float32)

        # Define Observation Space
        self.observation_space = spaces.Dict({
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(left_arm.dof + right_arm.dof,), dtype=np.float32),
            "images": spaces.Dict({
                "front": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "left_wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "right_wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "world": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
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
        self.task_actors = self.env._load_scene({}, task_name=self.task)
        # Reset robot pose to the cached "home" configuration
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        left_arm.set_qpos(self.init_qpos_left)
        right_arm.set_qpos(self.init_qpos_right)
        # Step simulation to settle
        for _ in range(10):
            self.scene.step()
        return self._get_obs(), {}

    def step(self, action):
        # Action is target joint positions
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        left_dof = left_arm.dof
        right_dof = right_arm.dof
        left_action = action[:left_dof]
        right_action = action[left_dof:left_dof+right_dof]
        for i, joint in enumerate(left_arm.get_active_joints()):
            joint.set_drive_target(left_action[i])
        for i, joint in enumerate(right_arm.get_active_joints()):
            joint.set_drive_target(right_action[i])
        # Step physics
        for _ in range(4):
            left_arm.set_qf(left_arm.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            right_arm.set_qf(right_arm.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.scene.step()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False # TODO: Define success condition
        reward = 0.0 # TODO: Define reward function
        return self._get_obs(), reward, terminated, truncated, {}


    def _get_obs(self):
        # Capture images using SO101TaskEnv's camera configs
        self.scene.update_render()
        # Get front camera (sensor_cam)
        front_img = None
        for cam in self.env._default_sensor_configs:
            cam_name = cam.name if hasattr(cam, 'name') else None
            if cam_name == "front_camera":
                # Find the camera entity in the scene
                for entity in self.scene.get_all_entities():
                    if hasattr(entity, 'name') and entity.name == "front_camera":
                        cam_comp = entity.get_component(RenderCameraComponent)
                        if cam_comp:
                            cam_comp.take_picture()
                            front_img = (cam_comp.get_picture("Color") * 255).astype(np.uint8)
        # Fallback: zeros if not found
        if front_img is None:
            front_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Wrist cameras
        left_wrist_cam = self.env.left_wrist_cam
        right_wrist_cam = self.env.right_wrist_cam
        left_wrist_cam.take_picture()
        left_wrist_img = (left_wrist_cam.get_picture("Color") * 255).astype(np.uint8)
        right_wrist_cam.take_picture()
        right_wrist_img = (right_wrist_cam.get_picture("Color") * 255).astype(np.uint8)

        # World camera (human_cam)
        world_img = None
        world_cam_cfg = self.env._default_human_render_camera_configs
        cam_name = world_cam_cfg.name if hasattr(world_cam_cfg, 'name') else None
        if cam_name == "world_demo_camera":
            for entity in self.scene.get_all_entities():
                if hasattr(entity, 'name') and entity.name == "world_demo_camera":
                    cam_comp = entity.get_component(RenderCameraComponent)
                    if cam_comp:
                        cam_comp.take_picture()
                        world_img = (cam_comp.get_picture("Color") * 255).astype(np.uint8)
        if world_img is None:
            world_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Proprioception
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        left_qpos = left_arm.get_qpos()
        right_qpos = right_arm.get_qpos()
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
        self.env.close()
