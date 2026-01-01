import numpy as np
import sapien
from transforms3d.quaternions import mat2quat
from lerobot.envs.sapien_env import SO101TaskEnv
from lerobot.common.motion_planning.so101.motionplanner import SO101ArmMotionPlanningSolver
from lerobot.common.motion_planning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb


CM = 0.01


def print_pose(name, pose: sapien.Pose):
    """Print position and quaternion of a pose."""
    p, q = pose.p, pose.q
    print(f"[{name}] pos = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  quat = {q}")


def build_grasp_pose(approaching, closing, center, finger_length):
    """
    Construct a grasp pose.
    TCP frame convention:
        Z = approach (tool forward), Y = finger closing, X = cross(Y, Z)
    """
    z_axis = approaching / np.linalg.norm(approaching)
    y_axis = closing - np.dot(closing, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = mat2quat(rot)

    # Position: retreat along approach direction by finger_length + 2cm
    tcp_pos = center - z_axis * (finger_length + 0.02)
    tcp_pos[2] = max(tcp_pos[2], 0.09)

    return sapien.Pose(tcp_pos, quat)


def solve_lift(env: SO101TaskEnv, seed=None, debug=False, vis=False):
    # env could be LeRobotGymEnv (with exposed agents) or direct SO101TaskEnv
    # If LeRobotGymEnv, it will have agent_right attribute; otherwise get from env.env
    if hasattr(env, 'agent_right'):
        actual_env = env
    else:
        actual_env = env.env  # Unwrap if needed
    
    actual_env.reset(seed=seed)
    agent = actual_env.agent_right  # SO101 agent instance with tcp_pose property
    robot = agent.robot      # ManiSkill Articulation for motion planning
    urdf_path = "assets/SO101/so101.urdf"
    tcp_link_name = "gripper_link"
    planner = SO101ArmMotionPlanningSolver(
        actual_env,
        robot,
        urdf_path,
        tcp_link_name,
        debug=False,
        vis=vis,
        base_pose=robot.get_root_pose(),
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    planner.open_gripper()
    # retrieves the object oriented bounding box (trimesh box object)
    cube = actual_env.task_actors[0]
    obb = get_actor_obb(cube)

    approaching = np.array([0, 0, -1])

    # rotate around x-axis to align with the expected frame for computing grasp poses (Z is up/down)
    # Use agent.tcp_pose.sp to get the sapien.Pose from ManiSkill Pose wrapper
    tcp_pose = sapien.Pose(q=np.array([0.70710678, 0, 0, 0.70710678])) * agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    # we can build a simple grasp pose using this information
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    # Use the agent's build_grasp_pose method which properly handles the grasp frame
    grasp_pose_computed = agent.build_grasp_pose(approaching, closing, cube.pose.sp.p)
    print(f"[DEBUG] cube.pose.sp.p = {cube.pose.sp.p}")
    print(f"[DEBUG] computed grasp_pose: p={grasp_pose_computed.p}, q={grasp_pose_computed.q}")
    
    # For SO101, use a simpler orientation: identity quaternion (straight down approach)
    # This is more likely to be within the reachable workspace
    grasp_pose = sapien.Pose(p=grasp_pose_computed.p, q=np.array([1, 0, 0, 0]))
    print(f"[DEBUG] grasp_pose (identity orientation): p={grasp_pose.p}, q={grasp_pose.q}")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    # Apply reaching offset in world coordinates, not in grasp frame
    # Reaching offset: [0, 0.02, 0.03] means 2cm towards cube, 3cm up from the table
    reaching_offset_pos = np.array([0, 0.02, 0.03])
    reach_pose = sapien.Pose(p=grasp_pose.p + reaching_offset_pos, q=grasp_pose.q)
    print(f"[DEBUG] reach_pose: p={reach_pose.p}, q={reach_pose.q}")
    # Ensure reach_pose is a sapien.Pose
    if not isinstance(reach_pose, sapien.Pose):
        reach_pose = sapien.Pose(reach_pose.p, reach_pose.q)
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper(t=12)

    # -------------------------------------------------------------------------- #
    # Move to lift pose
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.04]) * grasp_pose
    res = planner.move_to_pose_with_RRTConnect(lift_pose)
    planner.close()
    return res

    
# ================================================================
#                          STACK
# ================================================================
def solve_stack(env, seed=None, debug=False, vis=False):
    # env could be LeRobotGymEnv (with exposed agents) or direct SO101TaskEnv
    if hasattr(env, 'agent_right'):
        actual_env = env
    else:
        actual_env = env.env  # Unwrap if needed
    
    agent = actual_env.agent_right
    robot = agent.robot
    tcp_link_name = "gripper_link"

    planner = SO101ArmMotionPlanningSolver(
        actual_env, robot, "assets/SO101/so101.urdf", tcp_link_name,
        debug=debug, vis=vis,
        base_pose=robot.get_root_pose(),
        visualize_target_grasp_pose=vis,
        print_env_info=False
    )

    planner.open_gripper()
    FINGER_LENGTH = 0.025

    red_cube = actual_env.task_actors[0]
    obb_red = get_actor_obb(red_cube)

    approaching = np.array([0, 0, -1.0])

    tcp_pose = robot.find_link_by_name("gripper_link").pose.sp
    tcp_T = tcp_pose.to_transformation_matrix()
    target_closing = tcp_T[:3, 1]

    grasp_info = compute_grasp_info_by_obb(obb_red, approaching, target_closing, FINGER_LENGTH)

    grasp_pose = agent.build_grasp_pose(
        approaching,
        grasp_info["closing"],
        red_cube.pose.sp.p
    )

    reach_pose = sapien.Pose([0, 0, 0.10]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper(t=20)

    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(lift_pose)

    green_cube = actual_env.task_actors[1]
    place_pos = green_cube.pose.sp.p + np.array([0, 0, 0.035])
    place_pose = sapien.Pose(place_pos, grasp_pose.q)

    pre_place_pose = sapien.Pose([0, 0, 0.10]) * place_pose
    planner.move_to_pose_with_RRTConnect(pre_place_pose)
    planner.move_to_pose_with_RRTConnect(place_pose)
    planner.open_gripper(t=20)
    planner.move_to_pose_with_RRTConnect(pre_place_pose)
    planner.close()


# ================================================================
#                          SORT
# ================================================================
def solve_sort(env, seed=None, debug=False, vis=False):
    # env could be LeRobotGymEnv (with exposed agents) or direct SO101TaskEnv
    if hasattr(env, 'agent_right'):
        actual_env = env
    else:
        actual_env = env.env  # Unwrap if needed
    
    approaching = np.array([0, 0, -1.0])

    # ---------- Right arm ----------
    agent_r = actual_env.agent_right
    robot_r = agent_r.robot
    planner_r = SO101ArmMotionPlanningSolver(
        actual_env, robot_r, "assets/SO101/so101.urdf", "gripper_link",
        debug=debug, vis=vis,
        base_pose=robot_r.get_root_pose(),
        print_env_info=False
    )

    planner_r.open_gripper()
    red_cube = actual_env.task_actors[0]
    obb_red = get_actor_obb(red_cube)

    tcp_pose_r = robot_r.find_link_by_name("gripper_link").pose.sp
    tcp_T_r = tcp_pose_r.to_transformation_matrix()
    target_closing_r = tcp_T_r[:3, 1]

    grasp_info_r = compute_grasp_info_by_obb(obb_red, approaching, target_closing_r, 0.025)
    grasp_pose_r = agent_r.build_grasp_pose(approaching, grasp_info_r["closing"], red_cube.pose.sp.p)

    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.10]) * grasp_pose_r)
    planner_r.move_to_pose_with_RRTConnect(grasp_pose_r)
    planner_r.close_gripper(t=20)
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.15]) * grasp_pose_r)

    place_pose_r = sapien.Pose([0.571, 0.25, 0.02], grasp_pose_r.q)
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.10]) * place_pose_r)
    planner_r.move_to_pose_with_RRTConnect(place_pose_r)
    planner_r.open_gripper(t=20)
    planner_r.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.10]) * place_pose_r)
    planner_r.close()

    # ---------- Left arm ----------
    agent_l = actual_env.agent_left
    robot_l = agent_l.robot
    planner_l = SO101ArmMotionPlanningSolver(
        actual_env, robot_l, "assets/SO101/so101.urdf", "gripper_link",
        debug=debug, vis=vis,
        base_pose=robot_l.get_root_pose(),
        print_env_info=False
    )

    planner_l.open_gripper()
    green_cube = actual_env.task_actors[1]
    obb_green = get_actor_obb(green_cube)

    tcp_pose_l = robot_l.find_link_by_name("gripper_link").pose.sp
    tcp_T_l = tcp_pose_l.to_transformation_matrix()
    target_closing_l = tcp_T_l[:3, 1]

    grasp_info_l = compute_grasp_info_by_obb(obb_green, approaching, target_closing_l, 0.025)
    grasp_pose_l = agent_l.build_grasp_pose(approaching, grasp_info_l["closing"], green_cube.pose.sp.p)

    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.10]) * grasp_pose_l)
    planner_l.move_to_pose_with_RRTConnect(grasp_pose_l)
    planner_l.close_gripper(t=20)
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.15]) * grasp_pose_l)

    place_pose_l = sapien.Pose([0.029, 0.25, 0.02], grasp_pose_l.q)
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.10]) * place_pose_l)
    planner_l.move_to_pose_with_RRTConnect(place_pose_l)
    planner_l.open_gripper(t=20)
    planner_l.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.10]) * place_pose_l)
    planner_l.close()
