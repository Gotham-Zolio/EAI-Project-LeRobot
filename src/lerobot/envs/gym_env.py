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

        # Setup static scene (table/ground) only once
        self.env._load_scene({}, task_name=self.task)
        # Setup movable objects (cube, goal, etc.)
        self._reset_task_objects()

        # Cache a "home" joint configuration from the scene creation (defer to after reset)
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        # expose for planners
        self.left_arm = left_arm
        self.right_arm = right_arm
        # expose SO101 agents for task planners
        self.agent_left = self.env.agent_left
        self.agent_right = self.env.agent_right
        # task objects
        self.task_actors = [self.env.cube, self.env.goal_site]
        # get_qpos returns torch.Tensor; clone to avoid in-place mutation references
        self.init_qpos_left = left_arm.get_qpos().clone()
        self.init_qpos_right = right_arm.get_qpos().clone()

        # Compute DOF as ints (left_arm.dof may be torch.Tensor)
        self.left_dof = int(left_arm.dof.item()) if hasattr(left_arm.dof, "item") else int(left_arm.dof)
        self.right_dof = int(right_arm.dof.item()) if hasattr(right_arm.dof, "item") else int(right_arm.dof)
        total_dof = self.left_dof + self.right_dof

        # Define Action Space: actual DOF * 2 arms (e.g., 6+6=12 for SO-101)
        self.action_space = spaces.Box(low=-1, high=1, shape=(total_dof,), dtype=np.float32)

        # Define Observation Space
        self.observation_space = spaces.Dict({
            "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(total_dof,), dtype=np.float32),
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
        # Only randomize/reset movable objects, not table/ground
        self._reset_task_objects()
        # Reset robot pose to the cached "home" configuration
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        left_arm.set_qpos(self.init_qpos_left)
        right_arm.set_qpos(self.init_qpos_right)
        # Step simulation to settle
        for _ in range(10):
            self.scene.step()
        return self._get_obs(), {}

    def _reset_task_objects(self):
        # Reset cube, goal, etc. to random positions (matching pick_cube_so101.py _initialize_episode)
        # This assumes self.env has .cube, .goal_site, .cube_spawn_half_size, etc.
        import torch
        b = 1  # single env for gym.Env
        # Randomize cube position
        xyz = torch.zeros((b, 3))
        xyz[:, :2] = (
            torch.rand((b, 2)) * self.env.cube_spawn_half_size * 2
            - self.env.cube_spawn_half_size
        )
        xyz[:, 0] += self.env.cube_spawn_center[0]
        xyz[:, 1] += self.env.cube_spawn_center[1]
        xyz[:, 2] = self.env.cube_half_size
        from mani_skill.utils.structs.pose import Pose
        qs = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
        
        # Unbatch tensors for set_pose (they expect (3,) and (4,), not (1, 3) and (1, 4))
        cube_pose = Pose.create_from_pq(xyz[0], qs[0])
        self.env.cube.set_pose(cube_pose)
        
        # Randomize goal position
        goal_xyz = torch.zeros((b, 3))
        goal_xyz[:, :2] = (
            torch.rand((b, 2)) * self.env.cube_spawn_half_size * 2
            - self.env.cube_spawn_half_size
        )
        goal_xyz[:, 0] += self.env.cube_spawn_center[0]
        goal_xyz[:, 1] += self.env.cube_spawn_center[1]
        goal_xyz[:, 2] = torch.rand((b)) * self.env.max_goal_height + xyz[:, 2]
        
        # Unbatch for set_pose
        goal_pose = Pose.create_from_pq(goal_xyz[0], torch.tensor([1, 0, 0, 0], dtype=torch.float32))
        self.env.goal_site.set_pose(goal_pose)

    def step(self, action):
        # Action is target joint positions (2 arms concatenated)
        import torch
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        total_dof = self.left_dof + self.right_dof
        if action.shape[0] != total_dof:
            raise ValueError(f"Action length {action.shape[0]} != total dof {total_dof}")

        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        left_dof = self.left_dof
        right_dof = self.right_dof
        left_action = action[:left_dof]
        right_action = action[left_dof:left_dof+right_dof]
        for i, joint in enumerate(left_arm.get_active_joints()):
            joint.set_drive_target(np.array([float(left_action[i])], dtype=np.float32))
        for i, joint in enumerate(right_arm.get_active_joints()):
            joint.set_drive_target(np.array([float(right_action[i])], dtype=np.float32))
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
        def to_uint8(img, fallback_shape=(480, 640, 3)):
            if img is None:
                return np.zeros(fallback_shape, dtype=np.uint8)
            try:
                import torch
            except Exception:
                torch = None

            # Unwrap list/tuple (common camera return) by taking first element
            if isinstance(img, (list, tuple)) and len(img) > 0:
                img = img[0]

            # Torch tensor handling (including CUDA)
            if torch is not None and isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255 if arr.dtype.kind == 'f' else 1), 0, 255).astype(np.uint8)
            return arr

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
                            front_img = to_uint8(cam_comp.get_picture("Color"))
        # Fallback: zeros if not found
        if front_img is None:
            front_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Wrist cameras
        left_wrist_cam = self.env.left_wrist_cam
        right_wrist_cam = self.env.right_wrist_cam
        left_wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
        right_wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
        if left_wrist_cam is not None:
            try:
                left_wrist_cam.take_picture()
                left_wrist_img = to_uint8(left_wrist_cam.get_picture("Color"))
            except Exception as e:
                print(f"[WARN] left wrist cam capture failed: {e}")
        if right_wrist_cam is not None:
            try:
                right_wrist_cam.take_picture()
                right_wrist_img = to_uint8(right_wrist_cam.get_picture("Color"))
            except Exception as e:
                print(f"[WARN] right wrist cam capture failed: {e}")

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
                        world_img = to_uint8(cam_comp.get_picture("Color"))
        if world_img is None:
            world_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Proprioception
        left_arm = self.env.agent_left.robot
        right_arm = self.env.agent_right.robot
        left_qpos = left_arm.get_qpos()
        right_qpos = right_arm.get_qpos()
        # convert torch tensors to numpy
        if hasattr(left_qpos, "detach"):
            left_qpos = left_qpos.detach().cpu().numpy()
        if hasattr(right_qpos, "detach"):
            right_qpos = right_qpos.detach().cpu().numpy()
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
