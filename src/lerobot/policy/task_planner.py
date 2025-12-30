import numpy as np
import sapien
from transforms3d.quaternions import mat2quat
from lerobot.common.motion_planning.so101.motionplanner import SO101ArmMotionPlanningSolver
from lerobot.common.motion_planning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb


CM = 0.01

# 通用夹爪宽度调试函数
def get_gripper_width_debug(robot):
    # 优先调用get_gripper_width方法
    if hasattr(robot, 'get_gripper_width'):
        try:
            return f"Gripper width: {robot.get_gripper_width():.4f} m"
        except Exception as e:
            return f"get_gripper_width() error: {e}"
    # 尝试qpos属性
    if hasattr(robot, 'qpos'):
        try:
            qpos = robot.qpos
            # 假设夹爪宽度为qpos最后一个或前两个的差值（常见于双指夹爪）
            if isinstance(qpos, (list, np.ndarray)) and len(qpos) >= 2:
                width = abs(qpos[-1] - qpos[-2])
                return f"Gripper width (qpos): {width:.4f} (raw: {qpos})"
            else:
                return f"qpos: {qpos} (cannot infer width)"
        except Exception as e:
            return f"qpos error: {e}"
    # 尝试get_qpos方法
    if hasattr(robot, 'get_qpos'):
        try:
            qpos = robot.get_qpos()
            if isinstance(qpos, (list, np.ndarray)) and len(qpos) >= 2:
                width = abs(qpos[-1] - qpos[-2])
                return f"Gripper width (get_qpos): {width:.4f} (raw: {qpos})"
            else:
                return f"get_qpos: {qpos} (cannot infer width)"
        except Exception as e:
            return f"get_qpos() error: {e}"
    return "No gripper width method found."

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


def solve_lift(env, seed=None, debug=False, vis=False):
    """Plan and execute a lift task for the right arm in the given environment."""
    robot = env.right_arm
    urdf_path = "assets/SO101/so101.urdf"
    tcp_link_name = "gripper_link"

    print("\n==================== LIFT ====================")

    # --- Robot base ---
    base_pose = robot.get_root_pose()
    print_pose("right_arm base", base_pose)

    planner = SO101ArmMotionPlanningSolver(
        env, robot, urdf_path, tcp_link_name,
        debug=debug, vis=vis,
        base_pose=base_pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    planner.open_gripper(t=80)

    # --- Cube ---
    cube = env.task_actors[0]
    cube_pose = cube.pose
    print_pose("cube pose:", cube_pose)

    obb = get_actor_obb(cube)
    print("cube OBB:", obb)

    # --- TCP frame ---
    tcp_link = robot.find_link_by_name(tcp_link_name)
    tcp_pose = tcp_link.pose
    tcp_T = tcp_pose.to_transformation_matrix()

    # Use current TCP Y-axis as approach direction (gripper open/close direction)
    approaching = tcp_T[:3, 1].copy()  # Y axis is the true gripper approach
    # For closing, use current TCP Z-axis (orthogonal to Y)
    target_closing = tcp_T[:3, 2].copy()

    FINGER_LENGTH = 0.025

    # Compute grasp info using OBB and approach/closing directions
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_info["closing"] = target_closing.copy()

    # Build grasp pose: Y=approach, Z=closing, X=YxZ
    y_axis = approaching / np.linalg.norm(approaching)
    z_axis = target_closing - np.dot(target_closing, y_axis) * y_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    from scipy.spatial.transform import Rotation as R
    grasp_quat = R.from_matrix(rot).as_quat()
    grasp_pose_raw = sapien.Pose(grasp_info["center"], grasp_quat)

    # --- 获取cube与x轴夹角（yaw）并打印 ---
    from scipy.spatial.transform import Rotation as R
    q = cube_pose.q
    # sapien通常为[w, x, y, z]，scipy为[x, y, z, w]
    if len(q) == 4:
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_xyzw = np.array(q)
    r = R.from_quat(q_xyzw)
    yaw_deg = r.as_euler('zyx', degrees=True)[0]
    yaw_rad = r.as_euler('zyx', degrees=False)[0]
    print(f"[INFO] cube与x轴夹角: {yaw_deg:.2f}° ({yaw_rad:.3f} rad)")

    # --- Reach pose: move above grasp pose, 并绕z轴再转cube yaw角 ---
    reach_p = grasp_pose_raw.p.copy()
    reach_p[2] += 0.12  # Only move up in z direction
    reach_p[2] = max(reach_p[2], 0.09)
    dx = +0.000
    dy = -0.03
    reach_p[0] += dx
    reach_p[1] += dy
    print(f"[DEBUG] reach_p after offset: {reach_p}")
    # 先用原四元数，再绕z轴旋转yaw_rad
    # orig_rot = R.from_quat(grasp_pose_raw.q)
    # rot_z = R.from_euler('x', yaw_rad)
    # new_rot = rot_z * orig_rot
    # reach_quat = new_rot.as_quat()
    # reach_pose = sapien.Pose(reach_p, reach_quat)
    reach_pose = sapien.Pose(reach_p, grasp_pose_raw.q)
    print_pose("reach_pose", reach_pose)

    # --- Grasp pose: adjust to be slightly lower than reach ---
    grasp_pose_p = grasp_pose_raw.p.copy()
    grasp_pose_p[0] = reach_pose.p[0]
    grasp_pose_p[1] = reach_pose.p[1]
    grasp_pose_p[2] = reach_pose.p[2] - 0.09
    grasp_pose = sapien.Pose(grasp_pose_p, grasp_pose_raw.q)
    print(f"[DEBUG] grasp_pose.p: {grasp_pose.p}")

    # --- Lift pose: move up after grasping ---
    lift_pose_p = grasp_pose.p.copy()
    lift_pose_p[2] += 0.04
    lift_pose = sapien.Pose(lift_pose_p, grasp_pose.q)

    # Debug: check alignment between TCP Z-axis and grasp_pose Z-axis
    grasp_rot = R.from_quat(grasp_pose.q).as_matrix()
    cos_angle = np.dot(approaching, grasp_rot[:, 2]) / (np.linalg.norm(approaching) * np.linalg.norm(grasp_rot[:, 2]))
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    print(f"[Y check] cube Y = {cube_pose.p[1]:+.3f}, grasp Y = {grasp_pose.p[1]:+.3f}")

    # --- Motion execution ---
    planner.open_gripper(t=80)
    print(f"[DEBUG] Gripper width (before reach_pose): {get_gripper_width_debug(robot)}")
    planner.move_to_pose_with_RRTConnect(reach_pose, keep_gripper_open=True)
    print(f"[DEBUG] Gripper width (after reach_pose): {get_gripper_width_debug(robot)}")
    planner.move_to_pose_with_RRTConnect(grasp_pose, keep_gripper_open=True)
    print(f"[DEBUG] Gripper width (after grasp_pose): {get_gripper_width_debug(robot)}")
    planner.close_gripper(t=80)
    print(f"[DEBUG] Gripper width (after close_gripper): {get_gripper_width_debug(robot)}")
    planner.move_to_pose_with_RRTConnect(lift_pose)
    planner.close()

    print("====================================================\n")

    
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
