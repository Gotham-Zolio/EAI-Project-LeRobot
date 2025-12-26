
# ===== 新版：自动采样物块和target pose并可视化 gripper + cube =====
import numpy as np
import sapien
import os
import imageio
from scipy.spatial.transform import Rotation as R
from sapien.pysapien.render import RenderCameraComponent
from sapien import Entity, Pose

# 导入LeRobot环境和抓取规划器
from lerobot.envs.gym_env import LeRobotGymEnv
from lerobot.policy.task_planner import build_grasp_pose, get_actor_obb, compute_grasp_info_by_obb

def main():
    CM = 0.01

    # 1. 创建LeRobot环境并reset，采样物块
    env = LeRobotGymEnv(task="lift", headless=True, max_steps=1)
    env.reset()
    # 只用右臂和第一个物块
    robot = env.right_arm
    cube = env.task_actors[0]
    cube_pose = cube.pose


    # 2. 复用solve_lift的grasp pose采样逻辑
    tcp_link = robot.find_link_by_name("gripper_link")
    tcp_pose = tcp_link.pose
    tcp_T = tcp_pose.to_transformation_matrix()
    approaching = tcp_T[:3, 2].copy()
    target_closing = tcp_T[:3, 1].copy()
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(cube)
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_info["closing"] = target_closing  # enforce closing direction

    # 使夹爪Z轴（抓取方向）严格为世界-z方向，X轴为世界x正方向，Y轴自动正交
    z_axis = np.array([0, 0, -1.0])
    x_axis = np.array([1.0, 0, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)  # 保证正交
    grasp_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)
    # 额外绕z轴旋转，使夹爪Y轴与物块侧面平行
    cube_rot = R.from_quat(cube_pose.q)
    _, _, cube_z_angle = cube_rot.as_euler('xyz', degrees=False)
    extra_rot = R.from_euler('z', cube_z_angle)
    refined_rotmat = grasp_rotmat @ extra_rot.as_matrix()
    grasp_quat = R.from_matrix(refined_rotmat).as_quat()
    # 位置：cube中心沿z_axis（竖直向下）上移finger_length+0.02，最低不低于0.09m
    tcp_pos = grasp_info["center"] - z_axis * (FINGER_LENGTH + 0.02)
    tcp_pos[2] = max(tcp_pos[2], 0.09)
    grasp_pose = Pose(tcp_pos, grasp_quat)

    # 打印相关参数
    print("=== 可视化参数 ===")
    print(f"cube_pose.p: {[round(x, 5) for x in cube_pose.p]}")
    print(f"cube_pose.q: {[round(x, 5) for x in cube_pose.q]}")
    print(f"cube_size: {[2.5 * CM, 2.5 * CM, 2.5 * CM]}")
    print(f"grasp_pose.p: {[round(x, 5) for x in grasp_pose.p]}")
    print(f"grasp_pose.q: {[round(x, 5) for x in grasp_pose.q]}")
    print(f"gripper_urdf: assets/SO101/gripper_only.urdf")
    print("==================")

    # 输出夹爪坐标系X/Y/Z轴在世界坐标系下的方向
    rot = grasp_pose.to_transformation_matrix()[:3, :3]
    print("Gripper X (world):", [round(x, 5) for x in rot[:, 0]])
    print("Gripper Y (world):", [round(x, 5) for x in rot[:, 1]])
    print("Gripper Z (world):", [round(x, 5) for x in rot[:, 2]])

    # 3. 新建SAPIEN场景用于渲染
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0.3, 1, -0.3], [0.7, 0.7, 0.7])
    scene.add_directional_light([-0.3, 1, -0.1], [0.4, 0.4, 0.4])

    # 4. 加地面和边界线
    def add_box(scene, center, size, color):
        actor_builder = scene.create_actor_builder()
        half = [s / 2 for s in size]
        material = sapien.render.RenderMaterial()
        material.base_color = np.array(color)
        actor_builder.add_box_collision(half_size=half)
        actor_builder.add_box_visual(half_size=half, material=material)
        actor = actor_builder.build_static()
        actor.set_pose(sapien.Pose(center))
        return actor
    def add_floor(scene):
        width = 120 * CM
        height = 60 * CM
        thickness = 0.01
        builder = scene.create_actor_builder()
        half = [width / 2, height / 2, thickness / 2]
        material = sapien.render.RenderMaterial()
        material.base_color = np.array([0.92, 0.92, 0.92, 1])
        material.specular = 0.1
        builder.add_box_collision(half_size=half)
        builder.add_box_visual(half_size=half, material=material)
        floor = builder.build_static()
        floor.set_pose(sapien.Pose([width/2, height/2, -thickness/2]))
        return floor
    add_floor(scene)
    BORDER = 1.8 * CM
    d = 0.01 * CM
    black = [0, 0, 0, 1]
    add_box(scene, center=[60 * CM, 30 * CM, d / 2], size=[BORDER, 60 * CM, d], color=black)
    add_box(scene, center=[2.9 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[21.3 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[38.7 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[57.1 * CM, 25 * CM, d / 2], size=[BORDER, 16.4 * CM, d], color=black)
    add_box(scene, center=[21.3 * CM, 7.5 * CM, d / 2], size=[BORDER, 15 * CM, d], color=black)
    add_box(scene, center=[38.7 * CM, 7.5 * CM, d / 2], size=[BORDER, 15 * CM, d], color=black)
    add_box(scene, center=[30 * CM, 15.9 * CM, d / 2], size=[56 * CM, BORDER, d], color=black)
    add_box(scene, center=[30 * CM, 34.1 * CM, d / 2], size=[56 * CM, BORDER, d], color=black)

    # 5. 加载gripper-only URDF并设置target pose
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = False
    loader.load_multiple_visuals_from_file = True
    loader.scale = 1.0
    gripper_urdf = 'assets/SO101/gripper_only.urdf'
    gripper = loader.load(gripper_urdf)
    gripper.set_root_pose(grasp_pose)

    # 6. 可视化cube（物块）
    # cube_pose 已经有了，直接用 cube 的尺寸
    # 这里假设cube是2.5cm边长
    cube_size = [2.5 * CM, 2.5 * CM, 2.5 * CM]
    cube_color = [1.0, 0.0, 0.0, 1.0]
    add_box(scene, center=cube_pose.p, size=cube_size, color=cube_color)
    
    # 6.1 可视化夹爪自身坐标系（以grasp_pose为原点，X红Y绿Z蓝，长度2.5cm）
    def add_axis(scene, pose, length=0.08, thickness=0.003, arrow_size=0.012):
        T = pose.to_transformation_matrix()
        origin = pose.p
        # X轴-红
        x_dir = T[:3, 0]
        add_box(scene, center=origin + x_dir * length/2, size=[length, thickness, thickness], color=[1,0,0,1])
        add_box(scene, center=origin + x_dir * length, size=[arrow_size]*3, color=[1,0,0,1])
        # Y轴-绿
        y_dir = T[:3, 1]
        add_box(scene, center=origin + y_dir * length/2, size=[thickness, length, thickness], color=[0,1,0,1])
        add_box(scene, center=origin + y_dir * length, size=[arrow_size]*3, color=[0,1,0,1])
        # Z轴-蓝
        z_dir = T[:3, 2]
        add_box(scene, center=origin + z_dir * length/2, size=[thickness, thickness, length], color=[0,0,1,1])
        add_box(scene, center=origin + z_dir * length, size=[arrow_size]*3, color=[0,0,1,1])
    add_axis(scene, grasp_pose, length=0.08)

    # 7. 设置相机并渲染
    world_cam_mount = Entity()
    world_cam = RenderCameraComponent(width=800, height=600)
    near, far = 0.01, 50.0
    fx = 800 / (2 * np.tan(1.0 / 2))
    fy = fx
    cx = 800 / 2
    cy = 600 / 2
    world_cam.set_perspective_parameters(near, far, fx, fy, cx, cy, skew=0.0)
    world_cam_mount.add_component(world_cam)
    cam_x, cam_y, cam_z = -14.0 * CM, 60.0 * CM, 40.0 * CM
    quat = R.from_euler('xyz', [0.0, np.pi / 6, -np.pi / 4]).as_quat()
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    world_cam_mount.set_pose(Pose([cam_x, cam_y, cam_z], quat_sapien))
    scene.add_entity(world_cam_mount)

    # 8. 渲染并保存图片
    os.makedirs('logs/debug', exist_ok=True)
    scene.update_render()
    world_cam.take_picture()
    color = world_cam.get_picture("Color")
    img = (color * 255).astype(np.uint8)
    imageio.imwrite('logs/debug/gripper_pose.png', img)
    print('Saved gripper+cube visualization to logs/debug/gripper_pose.png')

if __name__ == '__main__':
    main()
