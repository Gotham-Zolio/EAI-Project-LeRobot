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

    env.reset(seed=seed)
    robot = env.right_arm
    urdf_path = "assets/SO101/so101.urdf"
    tcp_link_name = "gripper_link"
    planner = SO101ArmMotionPlanningSolver(
        env,
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
    cube = env.task_actors[0]
    obb = get_actor_obb(cube)

    approaching = np.array([0, 0, -1])

    # rotate around x-axis to align with the expected frame for computing grasp poses (Z is up/down)
    tcp_pose = sapien.Pose(q=np.array([0.70710678, 0, 0, 0.70710678])) * robot.find_link_by_name(tcp_link_name).pose
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    # we can build a simple grasp pose using this information
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    # Use the cube pose center for grasping
    grasp_pose = build_grasp_pose(approaching, closing, cube.pose.p, FINGER_LENGTH)

    # due to how SO101 is defined we may need to transform the grasp pose back to what is expected by SO101
    grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose([0, 0.02, 0.03]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose)
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
    robot = env.right_arm
    tcp_link_name = "gripper_link"

    planner = SO101ArmMotionPlanningSolver(
        env, robot, "assets/SO101/so101.urdf", tcp_link_name,
        debug=debug, vis=vis,
        base_pose=robot.get_root_pose(),
        visualize_target_grasp_pose=vis,
        print_env_info=False
    )

    planner.open_gripper()
    FINGER_LENGTH = 0.025

    red_cube = env.task_actors[0]
    obb_red = get_actor_obb(red_cube)

    approaching = np.array([0, 0, -1.0])

    tcp_pose = robot.find_link_by_name("gripper_link").pose
    tcp_T = tcp_pose.to_transformation_matrix()
    target_closing = tcp_T[:3, 1]

    grasp_info = compute_grasp_info_by_obb(obb_red, approaching, target_closing, FINGER_LENGTH)

    grasp_pose = build_grasp_pose(
        approaching,
        grasp_info["closing"],
        grasp_info["center"],
        FINGER_LENGTH
    )

    reach_pose = sapien.Pose([0, 0, 0.10]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper(t=20)

    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(lift_pose)

    green_cube = env.task_actors[1]
    place_pos = green_cube.pose.p + np.array([0, 0, 0.035])
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
    approaching = np.array([0, 0, -1.0])

    # ---------- Right arm ----------
    robot_r = env.right_arm
    planner_r = SO101ArmMotionPlanningSolver(
        env, robot_r, "assets/SO101/so101.urdf", "gripper_link",
        debug=debug, vis=vis,
        base_pose=robot_r.get_root_pose(),
        print_env_info=False
    )

    planner_r.open_gripper()
    red_cube = env.task_actors[0]
    obb_red = get_actor_obb(red_cube)

    tcp_pose_r = robot_r.find_link_by_name("gripper_link").pose
    tcp_T_r = tcp_pose_r.to_transformation_matrix()
    target_closing_r = tcp_T_r[:3, 1]

    grasp_info_r = compute_grasp_info_by_obb(obb_red, approaching, target_closing_r, 0.025)
    grasp_pose_r = build_grasp_pose(approaching, grasp_info_r["closing"], grasp_info_r["center"], 0.025)

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
    robot_l = env.left_arm
    planner_l = SO101ArmMotionPlanningSolver(
        env, robot_l, "assets/SO101/so101.urdf", "gripper_link",
        debug=debug, vis=vis,
        base_pose=robot_l.get_root_pose(),
        print_env_info=False
    )

    planner_l.open_gripper()
    green_cube = env.task_actors[1]
    obb_green = get_actor_obb(green_cube)

    tcp_pose_l = robot_l.find_link_by_name("gripper_link").pose
    tcp_T_l = tcp_pose_l.to_transformation_matrix()
    target_closing_l = tcp_T_l[:3, 1]

    grasp_info_l = compute_grasp_info_by_obb(obb_green, approaching, target_closing_l, 0.025)
    grasp_pose_l = build_grasp_pose(approaching, grasp_info_l["closing"], grasp_info_l["center"], 0.025)

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
