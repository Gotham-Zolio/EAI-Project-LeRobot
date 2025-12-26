import numpy as np
import sapien
from transforms3d.euler import euler2quat

from lerobot.common.motion_planning.so101.motionplanner import SO101ArmMotionPlanningSolver
from lerobot.common.motion_planning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

CM = 0.01

def solve_lift(env, seed=None, debug=False, vis=False):
    # env.reset(seed=seed) # Assumed to be reset by caller or we reset here?
    # Usually caller resets.
    
    robot = env.right_arm
    urdf_path = "assets/SO101/so101.urdf"
    tcp_link_name = "gripper_frame_link"
    
    planner = SO101ArmMotionPlanningSolver(
        env,
        robot,
        urdf_path,
        tcp_link_name,
        debug=debug,
        vis=vis,
        base_pose=robot.get_root_pose(),
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    planner.open_gripper()
    
    # Target: Red block
    cube = env.task_actors[0]
    obb = get_actor_obb(cube)

    approaching = np.array([0, 0, -1])

    # rotate around x-axis to align with the expected frame for computing grasp poses (Z is up/down)
    # We need the TCP pose in world frame to determine "target_closing" (y-axis of gripper?)
    # env.right_arm.find_link_by_name("gripper_link_tip").pose
    tcp_pose = robot.find_link_by_name(tcp_link_name).pose
    
    # Adjust for calculation
    tcp_pose_calc = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * tcp_pose
    target_closing = (tcp_pose_calc).to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    
    # Build grasp pose
    # We need a helper to build grasp pose from approach/closing/center.
    # ManiSkill agent had `build_grasp_pose`. We need to implement it or do it manually.
    # Rotation matrix from approach (z) and closing (y or x).
    # SO101: Z is approach? 
    # Let's look at `pick_cube.py`:
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    # grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    # We need to construct the rotation matrix.
    # approach is Z axis of gripper?
    # closing is Y axis?
    
    # Let's assume standard: Z approach, X closing (or Y).
    # If approach is [0,0,-1] (down), and closing is [0,1,0] (y).
    # X = Y x Z.
    
    from transforms3d.quaternions import mat2quat
    def build_grasp_pose(approaching, closing, center):
        # Construct rotation matrix: columns are x, y, z axes
        z_axis = approaching
        y_axis = closing - np.dot(closing, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3,3)
        quat = mat2quat(rot)  # [w, x, y, z]
        return sapien.Pose(center, quat)

    grasp_pose = build_grasp_pose(approaching, closing, center)
    
    # Apply the correction from pick_cube.py
    # grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    # Note: pick_cube.py used `env.agent.build_grasp_pose`.
    # If we use the manual construction, we might need to tune the rotation.
    # Let's try without correction first, or assume the construction matches.
    
    # Reach
    reach_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)
    
    # Grasp
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper(t=20)
    
    # Lift
    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(lift_pose)
    
    planner.close()

def solve_stack(env, seed=None, debug=False, vis=False):
    robot = env.right_arm
    urdf_path = "assets/SO101/so101.urdf"
    tcp_link_name = "gripper_frame_link"
    
    planner = SO101ArmMotionPlanningSolver(
        env, robot, urdf_path, tcp_link_name, debug=debug, vis=vis,
        base_pose=robot.get_root_pose(), visualize_target_grasp_pose=vis, print_env_info=False
    )

    FINGER_LENGTH = 0.025
    planner.open_gripper()
    
    # 1. Pick Red (task_actors[0])
    red_cube = env.task_actors[0]
    obb_red = get_actor_obb(red_cube)
    
    approaching = np.array([0, 0, -1])
    tcp_pose = robot.find_link_by_name(tcp_link_name).pose
    tcp_pose_calc = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * tcp_pose
    target_closing = (tcp_pose_calc).to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(obb_red, approaching, target_closing, FINGER_LENGTH)
    
    from transforms3d.quaternions import mat2quat
    def build_grasp_pose(approaching, closing, center):
        z_axis = approaching
        y_axis = closing - np.dot(closing, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)
        quat = mat2quat(rot)
        return sapien.Pose(center, quat)

    grasp_pose = build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])
    
    # Reach & Grasp Red
    reach_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper(t=20)
    
    # Lift Red
    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(lift_pose)
    
    # 2. Place on Green (task_actors[1])
    green_cube = env.task_actors[1]
    # We want to place Red ON TOP of Green.
    # Target position = Green position + Z offset (cube height).
    # Assuming cubes are ~3cm.
    green_pos = green_cube.pose.p
    place_pos = green_pos + np.array([0, 0, 0.035]) # Slightly higher than 3cm
    
    # We keep the same orientation as the grasp, or align with Green?
    # Keeping grasp orientation is easier.
    place_pose = sapien.Pose(place_pos, grasp_pose.q)
    
    # Move above place pose
    pre_place_pose = sapien.Pose([0, 0, 0.1]) * place_pose
    planner.move_to_pose_with_RRTConnect(pre_place_pose)
    
    # Move to place
    planner.move_to_pose_with_RRTConnect(place_pose)
    
    # Open gripper
    planner.open_gripper(t=20)
    
    # Move up
    planner.move_to_pose_with_RRTConnect(pre_place_pose)
    
    planner.close()

def solve_sort(env, seed=None, debug=False, vis=False):
    # Red -> Right Zone (Right Arm)
    # Green -> Left Zone (Left Arm)
    
    # --- Right Arm (Red) ---
    robot_r = env.right_arm
    planner_r = SO101ArmMotionPlanningSolver(
        env, robot_r, "assets/SO101/so101.urdf", "gripper_frame_link",
        debug=debug, vis=vis, base_pose=robot_r.get_root_pose(), print_env_info=False
    )
    
    planner_r.open_gripper()
    red_cube = env.task_actors[0]
    obb_red = get_actor_obb(red_cube)
    
    approaching = np.array([0, 0, -1])
    tcp_pose_r = robot_r.find_link_by_name("gripper_link_tip").pose
    tcp_pose_calc_r = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * tcp_pose_r
    target_closing_r = (tcp_pose_calc_r).to_transformation_matrix()[:3, 1]
    
    grasp_info_r = compute_grasp_info_by_obb(obb_red, approaching, target_closing_r, 0.025)
    
    from transforms3d.quaternions import mat2quat
    def build_grasp_pose(approaching, closing, center):
        z_axis = approaching
        y_axis = closing - np.dot(closing, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)
        quat = mat2quat(rot)
        return sapien.Pose(center, quat)

    grasp_pose_r = build_grasp_pose(approaching, grasp_info_r["closing"], grasp_info_r["center"])
    
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.05]) * grasp_pose_r)
    planner_r.move_to_pose_with_RRTConnect(grasp_pose_r)
    planner_r.close_gripper(t=20)
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.15]) * grasp_pose_r)
    
    # Place Right (Red)
    # Right Zone: [57.1 * CM, 25 * CM] -> [0.571, 0.25]
    place_pos_r = np.array([0.571, 0.25, 0.02])
    place_pose_r = sapien.Pose(place_pos_r, grasp_pose_r.q)
    
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.1]) * place_pose_r)
    planner_r.move_to_pose_with_RRTConnect(place_pose_r)
    planner_r.open_gripper(t=20)
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.1]) * place_pose_r)
    planner_r.close()
    
    # --- Left Arm (Green) ---
    robot_l = env.left_arm
    planner_l = SO101ArmMotionPlanningSolver(
        env, robot_l, "assets/SO101/so101.urdf", "gripper_frame_link",
        debug=debug, vis=vis, base_pose=robot_l.get_root_pose(), print_env_info=False
    )
    
    planner_l.open_gripper()
    green_cube = env.task_actors[1]
    obb_green = get_actor_obb(green_cube)
    
    tcp_pose_l = robot_l.find_link_by_name("gripper_link_tip").pose
    tcp_pose_calc_l = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * tcp_pose_l
    target_closing_l = (tcp_pose_calc_l).to_transformation_matrix()[:3, 1]
    
    grasp_info_l = compute_grasp_info_by_obb(obb_green, approaching, target_closing_l, 0.025)
    grasp_pose_l = build_grasp_pose(approaching, grasp_info_l["closing"], grasp_info_l["center"])
    
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.05]) * grasp_pose_l)
    planner_l.move_to_pose_with_RRTConnect(grasp_pose_l)
    planner_l.close_gripper(t=20)
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.15]) * grasp_pose_l)
    
    # Place Left (Green)
    # Left Zone: [2.9 * CM, 25 * CM] -> [0.029, 0.25]
    place_pos_l = np.array([0.029, 0.25, 0.02])
    place_pose_l = sapien.Pose(place_pos_l, grasp_pose_l.q)
    
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.1]) * place_pose_l)
    planner_l.move_to_pose_with_RRTConnect(place_pose_l)
    planner_l.open_gripper(t=20)
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.1]) * place_pose_l)
    planner_l.close()

