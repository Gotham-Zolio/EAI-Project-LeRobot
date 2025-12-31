from typing import Any, Dict, Union
import numpy as np
import sapien
from sapien import Entity
from sapien.pysapien.render import RenderCameraComponent
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R
from lerobot.common.camera import (
    FRONT_CAM_W, FRONT_CAM_H, FRONT_FX, FRONT_FY, FRONT_CX, FRONT_CY
)
# === ManiSkill imports for agent/robot management ===
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from lerobot.agents.robots.so101.so_101 import SO101


# ---------------- Global Constants ----------------
CM = 0.01

# ---------------- Utility / Scene building ----------------


# =================== ManiSkill-style Env ===================
class SO101TaskEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["so101_left", "so101_right"]
    agent_left: SO101
    agent_right: SO101
    robot_init_qpos_noise = 0.02
    # front_cam (sensor_cam)
    sensor_cam_eye_pos = np.array([0.316, 0.26, 0.407])
    sensor_cam_target_pos = np.array([0.316, 0.26, 0.0])
    # world demo camera (human_cam)
    human_cam_eye_pos = np.array([0.5, 0.55, 0.15])
    human_cam_target_pos = np.array([0.5, 0.25, 0.15])
    max_goal_height = 0.07
    lock_z = True

    def __init__(self, *args, robot_uids="so101", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # pick_cube style front camera config
        front_cam_pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig(
            "front_camera", front_cam_pose, 640, 480, np.deg2rad(50), 0.01, 50.0
        )]

    @property
    def _default_human_render_camera_configs(self):
        # pick_cube style world demo camera config
        world_cam_pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig(
            "world_demo_camera", world_cam_pose, 640, 480, np.deg2rad(50), 0.01, 50.0
        )

    def _load_scene(self, options: dict):
        # pick_cube_so101.py style: TableSceneBuilder for table/boundaries, then build cube, goal, and spawn region
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        # Visualize cube spawn region (thin blue box, non-colliding)
        self.cube_spawn_center = getattr(self, 'cube_spawn_center', (0.48, 0.25))
        self.cube_half_size = getattr(self, 'cube_half_size', 0.015)
        self.cube_spawn_half_size = getattr(self, 'cube_spawn_half_size', 0.08)
        spawn_center = [self.cube_spawn_center[0], self.cube_spawn_center[1], self.cube_half_size]
        spawn_half_size = [self.cube_spawn_half_size, self.cube_spawn_half_size, 1e-4]
        self.cube_spawn_vis = actors.build_box(
            self.scene,
            half_sizes=spawn_half_size,
            color=[0, 0, 1, 0.2],
            name="cube_spawn_region",
            add_collision=False,
            initial_pose=sapien.Pose(p=spawn_center),
        )
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_thresh = getattr(self, 'goal_thresh', 0.01875)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects = getattr(self, '_hidden_objects', [])
        self._hidden_objects.append(self.goal_site)


    # TableSceneBuilder now handles boundaries and table, so this is no longer needed.

    def _load_agent(self, options: dict):
        # 采用 ManiSkill/pick_cube_so101.py 风格，调用父类自动实例化 agent
        # 左臂和右臂初始位姿可根据实际需求调整
        super()._load_agent(options)
        # 给左右臂分别添加腕部摄像头（如有需要）
        if hasattr(self, 'agent_left') and hasattr(self.agent_left, 'robot'):
            self.left_wrist_cam = self.add_wrist_camera(self.agent_left.robot)
        if hasattr(self, 'agent_right') and hasattr(self.agent_right, 'robot'):
            self.right_wrist_cam = self.add_wrist_camera(self.agent_right.robot)

    def initialize_agent(self, env_idx):
        b = len(env_idx)
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        qpos_left = (
            self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
            + qpos
        )
        qpos_right = (
            self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
            + qpos
        )
        self.agent_left.reset(qpos_left)
        left_pose = torch.tensor([0.119, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        right_pose = torch.tensor([0.481, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        assert left_pose.shape == (7,), f"left_pose shape must be (7,), got {left_pose.shape}"
        assert right_pose.shape == (7,), f"right_pose shape must be (7,), got {right_pose.shape}"
        self.agent_left.robot.set_pose(Pose(left_pose))
        self.agent_right.reset(qpos_right)
        self.agent_right.robot.set_pose(Pose(right_pose))

    # _load_scene_lift is not needed; pick_cube_so101.py does not use it. All scene setup is in _load_scene.

    def add_goal_site(self, scene, center, radius, color):
        # Use ManiSkill's actors.build_sphere for goal site
        actor = actors.build_sphere(
            scene,
            radius=radius,
            color=color,
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(center),
        )
        return actor

    def load_scene(self, scene, task_name, rng=None):
        if task_name == "default":
            return []
        elif task_name == "lift":
            return self._load_scene_lift(scene, rng=rng)
        elif task_name == "sort":
            return self._load_scene_sort(scene, rng=rng)
        elif task_name == "stack":
            return self._load_scene_stack(scene, rng=rng)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
    def add_wrist_camera(self, robot, link_name="camera_link", fovy_deg=50.0, z_offset=0.05, near=0.01, far=5.0):
        """
        Attach a RenderCameraComponent to a link (wrist). Use set_local_pose so it
        respects SAPIEN 3.x API (no set_world_pose for RenderCameraComponent).
        Returns the camera component.
        """
        link = robot.find_link_by_name(link_name)
        if link is None:
            raise ValueError(f"Link named '{link_name}' not found on robot")

        cam_w = 640
        cam_h = 480
        fovy = np.deg2rad(fovy_deg)
        fx = cam_w / (2 * np.tan(fovy / 2))
        fy = fx
        cx = cam_w / 2
        cy = cam_h / 2

        cam = RenderCameraComponent(width=cam_w, height=cam_h)
        cam.set_perspective_parameters(near, far, fx, fy, cx, cy, skew=0.0)

        link.entity.add_component(cam)

        offset = np.array([0.0, 0.0, z_offset], dtype=np.float32)
        quat = R.from_euler('xyz', [-np.pi/2, 0.0, 0.0]).as_quat()  # xyzw
        quat_sapien = [quat[3], quat[0], quat[1], quat[2]]         # wxyz

        cam.set_local_pose(sapien.Pose(offset, quat_sapien))
        return cam


# # ---------------- Scene configurations ----------------
# def get_random_pose(x_range, y_range, z_height, rng=None):
#     if rng is None:
#         x = np.random.uniform(*x_range)
#         y = np.random.uniform(*y_range)
#         rot_z = np.random.uniform(np.pi / 4, np.pi / 2)
#     else:
#         x = rng.uniform(*x_range)
#         y = rng.uniform(*y_range)
#         rot_z = rng.uniform(np.pi / 4, np.pi / 2)
#     z = z_height
#     return [x, y, z], rot_z

# def setup_scene_lift(scene, rng=None):
#     """Task Lift: one red block in the rightmost box"""
#     # Optimized spawn range for better arm reachability
#     # X: 44~47cm (reduced from 50cm - high X values cause IK failures even with good Y)
#     # Y: 22~24.5cm (safe range verified by testing)
#     # Test data: X>48cm causes IK failures or growing offsets; X~44-47cm gives 12-15cm initial offset
#     pos, rot = get_random_pose([44.0 * CM, 47.0 * CM], [22.0 * CM, 24.5 * CM], 1.5 * CM, rng=rng)
#     actor = add_block(scene, center=pos, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot)
#     return [actor]


# def setup_scene_stack(scene, rng=None):
#     """Task Sort: red + green in rightmost box"""
#     # x in 41.1~54.7cm, y in 18.3~31.7cm, dist >= 4.5cm
#     while True:
#         pos1, rot1 = get_random_pose([41.1 * CM, 54.7 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM, rng=rng)
#         pos2, rot2 = get_random_pose([41.1 * CM, 54.7 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM, rng=rng)
#         dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
#         if dist >= 4.5 * CM:
#             break
            
#     a1 = add_block(scene, center=pos1, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot1)
#     a2 = add_block(scene, center=pos2, color=[0.0, 0.8, 0.0, 1.0], label="Green", rotation_z=rot2)
#     return [a1, a2]


# def setup_scene_sort(scene, rng=None):
#     """Task Stack: red + green in the middle box"""
#     # x in 23.7~36.3cm, y in 18.3~31.7cm, dist >= 4.5cm
#     while True:
#         pos1, rot1 = get_random_pose([23.7 * CM, 36.3 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM, rng=rng)
#         pos2, rot2 = get_random_pose([23.7 * CM, 36.3 * CM], [18.3 * CM, 31.7 * CM], 1.5 * CM, rng=rng)
#         dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
#         if dist >= 4.5 * CM:
#             break

#     a1 = add_block(scene, center=pos1, color=[1.0, 0.0, 0.0, 1.0], label="Red", rotation_z=rot1)
#     a2 = add_block(scene, center=pos2, color=[0.0, 0.8, 0.0, 1.0], label="Green", rotation_z=rot2)
#     return [a1, a2]



# def setup_scene_stack(
#     scene,
#     cube_half_size=0.015,
#     goal_thresh=0.01875,
#     cube_spawn_half_size=0.08,
#     cube_spawn_center=(0.48, 0.25),
#     max_goal_height=0.07,
#     rng=None,
#     visualize_spawn=True,
#     lock_z=True,
# ):
#     """
#     Setup a stack task scene: two cubes (red, green) in spawn region, two goal sites.
#     Returns: [cube_actors], [goal_site_actors], (optional) cube_spawn_vis
#     """
#     if rng is None:
#         rng = np.random
#     cube_spawn_vis = None
#     if visualize_spawn:
#         spawn_center = [cube_spawn_center[0], cube_spawn_center[1], cube_half_size]
#         cube_spawn_vis = add_box(
#             scene,
#             center=spawn_center,
#             size=[2 * cube_spawn_half_size, 2 * cube_spawn_half_size, 2e-4],
#             color=[0, 0, 1, 0.2],
#         )
#     # 采样两个cube位置，保证距离大于cube边长
#     while True:
#         xy1 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2)
#         xy2 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2)
#         xy1[0] += cube_spawn_center[0]
#         xy1[1] += cube_spawn_center[1]
#         xy2[0] += cube_spawn_center[0]
#         xy2[1] += cube_spawn_center[1]
#         if np.linalg.norm(xy1 - xy2) >= 2 * cube_half_size:
#             break
#     z = cube_half_size
#     rot1 = rng.uniform(0, 2 * np.pi) if lock_z else 0.0
#     rot2 = rng.uniform(0, 2 * np.pi) if lock_z else 0.0
#     cube1 = add_block(
#         scene,
#         center=[xy1[0], xy1[1], z],
#         color=[1, 0, 0, 1],
#         label="red",
#         rotation_z=rot1,
#         size=[2 * cube_half_size] * 3,
#     )
#     cube2 = add_block(
#         scene,
#         center=[xy2[0], xy2[1], z],
#         color=[0, 0.8, 0, 1],
#         label="green",
#         rotation_z=rot2,
#         size=[2 * cube_half_size] * 3,
#     )
#     # 采样两个goal位置
#     goal_xy1 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2) + np.array(cube_spawn_center)
#     goal_xy2 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2) + np.array(cube_spawn_center)
#     goal_z1 = rng.uniform(0, max_goal_height) + z
#     goal_z2 = rng.uniform(0, max_goal_height) + z
#     goal1 = add_goal_site(
#         scene,
#         center=[goal_xy1[0], goal_xy1[1], goal_z1],
#         radius=goal_thresh,
#         color=[0, 1, 0, 1],
#     )
#     goal2 = add_goal_site(
#         scene,
#         center=[goal_xy2[0], goal_xy2[1], goal_z2],
#         radius=goal_thresh,
#         color=[0, 1, 0, 1],
#     )
#     if visualize_spawn:
#         return [cube1, cube2], [goal1, goal2], cube_spawn_vis
#     else:
#         return [cube1, cube2], [goal1, goal2]

# def setup_scene_sort(
#     scene,
#     cube_half_size=0.015,
#     goal_thresh=0.01875,
#     cube_spawn_half_size=0.08,
#     cube_spawn_center=(0.24, 0.25),
#     max_goal_height=0.07,
#     rng=None,
#     visualize_spawn=True,
#     lock_z=True,
# ):
#     """
#     Setup a sort task scene: two cubes (red, green) in spawn region, two goal sites.
#     Returns: [cube_actors], [goal_site_actors], (optional) cube_spawn_vis
#     """
#     if rng is None:
#         rng = np.random
#     cube_spawn_vis = None
#     if visualize_spawn:
#         spawn_center = [cube_spawn_center[0], cube_spawn_center[1], cube_half_size]
#         cube_spawn_vis = add_box(
#             scene,
#             center=spawn_center,
#             size=[2 * cube_spawn_half_size, 2 * cube_spawn_half_size, 2e-4],
#             color=[0, 0, 1, 0.2],
#         )
#     # 采样两个cube位置，保证距离大于cube边长
#     while True:
#         xy1 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2)
#         xy2 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2)
#         xy1[0] += cube_spawn_center[0]
#         xy1[1] += cube_spawn_center[1]
#         xy2[0] += cube_spawn_center[0]
#         xy2[1] += cube_spawn_center[1]
#         if np.linalg.norm(xy1 - xy2) >= 2 * cube_half_size:
#             break
#     z = cube_half_size
#     rot1 = rng.uniform(0, 2 * np.pi) if lock_z else 0.0
#     rot2 = rng.uniform(0, 2 * np.pi) if lock_z else 0.0
#     cube1 = add_block(
#         scene,
#         center=[xy1[0], xy1[1], z],
#         color=[1, 0, 0, 1],
#         label="red",
#         rotation_z=rot1,
#         size=[2 * cube_half_size] * 3,
#     )
#     cube2 = add_block(
#         scene,
#         center=[xy2[0], xy2[1], z],
#         color=[0, 0.8, 0, 1],
#         label="green",
#         rotation_z=rot2,
#         size=[2 * cube_half_size] * 3,
#     )
#     # 采样两个goal位置
#     goal_xy1 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2) + np.array(cube_spawn_center)
#     goal_xy2 = rng.uniform(-cube_spawn_half_size, cube_spawn_half_size, size=2) + np.array(cube_spawn_center)
#     goal_z1 = rng.uniform(0, max_goal_height) + z
#     goal_z2 = rng.uniform(0, max_goal_height) + z
#     goal1 = add_goal_site(
#         scene,
#         center=[goal_xy1[0], goal_xy1[1], goal_z1],
#         radius=goal_thresh,
#         color=[0, 1, 0, 1],
#     )
#     goal2 = add_goal_site(
#         scene,
#         center=[goal_xy2[0], goal_xy2[1], goal_z2],
#         radius=goal_thresh,
#         color=[0, 1, 0, 1],
#     )
#     if visualize_spawn:
#         return [cube1, cube2], [goal1, goal2], cube_spawn_vis
#     else:
#         return [cube1, cube2], [goal1, goal2]