import numpy as np
import sapien
from transforms3d.quaternions import mat2quat
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


    # ========== 可视化辅助函数 ========== 
    def add_axis_to_scene(scene, pose, length=0.08, thickness=0.003, arrow_size=0.012):
        def add_box(offset, size, color):
            builder = scene.create_actor_builder()
            half = [s / 2 for s in size]
            import sapien.render
            material = sapien.render.RenderMaterial()
            material.base_color = np.array(color)
            material.specular = 0.1
            builder.add_box_visual(half_size=half, material=material)
            actor = builder.build_static()
            # 以父pose为基础，沿主轴方向平移offset，姿态与父pose一致
            T = pose.to_transformation_matrix()
            new_pos = pose.p + T[:3, :3] @ offset
            actor.set_pose(sapien.Pose(new_pos, pose.q))
            return actor
        # X轴-红
        add_box([length/2, 0, 0], [length, thickness, thickness], [1,0,0,1])
        add_box([length, 0, 0], [arrow_size]*3, [1,0,0,1])
        # Y轴-绿
        add_box([0, length/2, 0], [thickness, length, thickness], [0,1,0,1])
        add_box([0, length, 0], [arrow_size]*3, [0,1,0,1])
        # Z轴-蓝
        add_box([0, 0, length/2], [thickness, thickness, length], [0,0,1,1])
        add_box([0, 0, length], [arrow_size]*3, [0,0,1,1])

    planner.open_gripper(t=80)



    # --- Cube ---
    cube = env.task_actors[0]
    
    '''
    # =========================================================
    # 【安全版】魔法物理属性修改 (自动查找刚体组件)
    # =========================================================
    try:
        print("Attempting to modify physics properties...")
        rigid_component = None
        # 遍历所有组件，寻找看起来像刚体的组件(有mass属性)
        for comp in cube.components:
            if hasattr(comp, 'mass'):
                rigid_component = comp
                break
        
        if rigid_component:
            # 1. 修改质量
            print(f"Found rigid component: {type(rigid_component).__name__}")
            rigid_component.mass = 0.01  # 变轻
            print(" -> Mass set to 0.01")
            
            # 2. 修改摩擦力
            if hasattr(env, 'scene') and hasattr(rigid_component, 'collision_shapes'):
                # static, dynamic, restitution
                new_mat = env.scene.create_physical_material(50.0, 50.0, 0.0)
                for shape in rigid_component.collision_shapes:
                    shape.set_physical_material(new_mat)
                print(" -> Friction set to 50.0")
        else:
            print("Warning: No rigid body component found on cube. Skipping physics tweak.")
            
    except Exception as e:
        print(f"Non-fatal error modifying physics: {e}")
        print("Continuing anyway...")
    # =========================================================

    '''

    cube_pose = cube.pose
    print_pose("cube pose", cube_pose)




    obb = get_actor_obb(cube)

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

    # --- 获取cube与x轴夹角（yaw) ---
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

    # --- Grasp pose: rotate grasp_pose around world Z by -yaw_rad + 90deg, offset along cube X and Y ---
    def wxyz_to_xyzw(q):
        return [q[1], q[2], q[3], q[0]]
    grasp_rot = R.from_quat(wxyz_to_xyzw(grasp_pose_raw.q))
    rot_y = R.from_euler('y', yaw_rad - np.pi / 2, degrees=False)
    new_rot = grasp_rot * rot_y
    new_quat = new_rot.as_quat()  # xyzw
    new_quat_wxyz = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
    grasp_p = cube_pose.p + np.array([0.0, 0.0, 0.056]) #- np.array([0.0, 0.0, 0.0]) #改之前：0.062, 0.024
    cube_rot = R.from_quat(wxyz_to_xyzw(cube_pose.q))
    cube_x_axis = cube_rot.apply([1, 0, 0])  # 世界系下的x轴方向
    grasp_p = grasp_p - 0.024 * cube_x_axis
    grasp_pose = sapien.Pose(grasp_p, new_quat_wxyz)
    print_pose("grasp pose", grasp_pose)

    # --- Lift pose: strictly move from grasp_pose along world +z by 0.05, keep orientation ---
    lift_pose_p = grasp_pose.p + np.array([0.0, 0.0, 0.05])
    lift_pose = sapien.Pose(lift_pose_p, grasp_pose.q)
    print_pose("lift pose", lift_pose)

    # --- Visualize axes ---
    if vis and hasattr(env, 'scene'):
        scene = env.scene
        add_axis_to_scene(scene, cube_pose, length=0.08)
        add_axis_to_scene(scene, grasp_pose, length=0.08)

    # --- Motion execution ---
    planner.open_gripper(t=80)
    print(f"\n1. Executing motion to grasp the cube...")
    planner.move_to_pose_with_RRTConnect(grasp_pose, keep_gripper_open=True)
    print(f"2. Closing gripper to grasp the cube...")
    planner.close_gripper(t=80, gap=0.6)
    print(f"3. Lifting the cube...")
    planner.move_to_pose_with_RRTConnect(lift_pose)
    print(f"4. Lift task completed.")
    planner.close()

    print("====================================================\n")

    
# ================================================================
#                          STACK
# ================================================================
def solve_stack(env, seed=None, debug=False, vis=False):
    robot = env.right_arm
    urdf_path = "assets/SO101/so101.urdf"
    tcp_link_name = "gripper_link"

    print("\n==================== STACK ====================")

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

    # ========== 可视化辅助函数 ========== 
    def add_axis_to_scene(scene, pose, length=0.08, thickness=0.003, arrow_size=0.012):
        def add_box(offset, size, color):
            builder = scene.create_actor_builder()
            half = [s / 2 for s in size]
            import sapien.render
            material = sapien.render.RenderMaterial()
            material.base_color = np.array(color)
            material.specular = 0.1
            builder.add_box_visual(half_size=half, material=material)
            actor = builder.build_static()
            # 以父pose为基础，沿主轴方向平移offset，姿态与父pose一致
            T = pose.to_transformation_matrix()
            new_pos = pose.p + T[:3, :3] @ offset
            actor.set_pose(sapien.Pose(new_pos, pose.q))
            return actor
        # X轴-红
        add_box([length/2, 0, 0], [length, thickness, thickness], [1,0,0,1])
        add_box([length, 0, 0], [arrow_size]*3, [1,0,0,1])
        # Y轴-绿
        add_box([0, length/2, 0], [thickness, length, thickness], [0,1,0,1])
        add_box([0, length, 0], [arrow_size]*3, [0,1,0,1])
        # Z轴-蓝
        add_box([0, 0, length/2], [thickness, thickness, length], [0,0,1,1])
        add_box([0, 0, length], [arrow_size]*3, [0,0,1,1])

    planner.open_gripper(t=80)

    # --- Red Cube (Target) ---
    red_cube = env.task_actors[0]
    red_cube_pose = red_cube.pose
    print_pose("red_cube pose", red_cube_pose)
    obb_red = get_actor_obb(red_cube)

    # --- Green Cube (Base) ---
    green_cube = env.task_actors[1]
    green_cube_pose = green_cube.pose
    print_pose("green_cube pose", green_cube_pose)

    # --- TCP frame ---
    tcp_link = robot.find_link_by_name(tcp_link_name)
    tcp_pose = tcp_link.pose
    tcp_T = tcp_pose.to_transformation_matrix()

    # Use current TCP Y-axis as approach direction (gripper open/close direction)
    approaching = tcp_T[:3, 1].copy()
    # For closing, use current TCP Z-axis (orthogonal to Y)
    target_closing = tcp_T[:3, 2].copy()

    FINGER_LENGTH = 0.025

    # Compute grasp info using OBB and approach/closing directions
    grasp_info = compute_grasp_info_by_obb(
        obb_red,
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

    # --- 获取cube与x轴夹角（yaw) ---
    q = red_cube_pose.q
    if len(q) == 4:
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_xyzw = np.array(q)
    r = R.from_quat(q_xyzw)
    yaw_deg = r.as_euler('zyx', degrees=True)[0]
    yaw_rad = r.as_euler('zyx', degrees=False)[0]

    # --- Grasp pose: rotate grasp_pose around world Z by -yaw_rad + 90deg, offset along cube X and Y ---
    def wxyz_to_xyzw(q):
        return [q[1], q[2], q[3], q[0]]
    grasp_rot = R.from_quat(wxyz_to_xyzw(grasp_pose_raw.q))
    rot_y = R.from_euler('y', yaw_rad - np.pi / 2, degrees=False)
    new_rot = grasp_rot * rot_y
    new_quat = new_rot.as_quat()  # xyzw
    new_quat_wxyz = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
    
    grasp_p = red_cube_pose.p + np.array([0.0, 0.0, 0.062])
    cube_rot = R.from_quat(wxyz_to_xyzw(red_cube_pose.q))
    cube_x_axis = cube_rot.apply([1, 0, 0])  # 世界系下的x轴方向
    grasp_p = grasp_p - 0.024 * cube_x_axis
    grasp_pose = sapien.Pose(grasp_p, new_quat_wxyz)
    print_pose("grasp pose", grasp_pose)

    # --- Lift pose ---
    lift_pose_p = grasp_pose.p + np.array([0.0, 0.0, 0.05])
    lift_pose = sapien.Pose(lift_pose_p, grasp_pose.q)
    print_pose("lift pose", lift_pose)

    # --- Place pose ---
    # Place on top of green cube.
    cube_height = 0.04
    target_red_cube_p = green_cube_pose.p + np.array([0, 0, cube_height])
    target_red_cube_q = green_cube_pose.q # Align orientation
    target_red_cube_pose = sapien.Pose(target_red_cube_p, target_red_cube_q)
    
    # Calculate gripper pose relative to red cube
    T_red_gripper = red_cube_pose.inv() * grasp_pose
    
    # Target gripper pose
    place_pose = target_red_cube_pose * T_red_gripper
    print_pose("place pose", place_pose)
    
    # Pre-place pose (higher)
    # Reduced offset from 0.10 to 0.05 to avoid IK issues at high Z
    pre_place_pose = sapien.Pose(place_pose.p + np.array([0, 0, 0.05]), place_pose.q)

    # --- Visualize axes ---
    if vis and hasattr(env, 'scene'):
        scene = env.scene
        add_axis_to_scene(scene, red_cube_pose, length=0.08)
        add_axis_to_scene(scene, green_cube_pose, length=0.08)
        add_axis_to_scene(scene, grasp_pose, length=0.08)
        add_axis_to_scene(scene, place_pose, length=0.08)

    # --- Motion execution ---
    print(f"\n1. Executing motion to grasp the red cube...")
    # Add a pre-grasp pose to help the planner
    pre_grasp_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, 0.05]), grasp_pose.q)
    planner.move_to_pose_with_RRTConnect(pre_grasp_pose, keep_gripper_open=True)
    planner.move_to_pose_with_RRTConnect(grasp_pose, keep_gripper_open=True)
    
    print(f"2. Closing gripper to grasp the red cube...")
    # gap=0.8 means 80% open! Use 0.1 for tight grasp.
    planner.close_gripper(t=80, gap=0.6) 
    
    print(f"3. Lifting the red cube...")
    planner.move_to_pose_with_RRTConnect(lift_pose)
    
    print(f"4. Moving to place position...")
    planner.move_to_pose_with_RRTConnect(pre_place_pose)
    planner.move_to_pose_with_RRTConnect(place_pose)
    
    print(f"5. Opening gripper to release...")
    planner.open_gripper(t=80)
    
    print(f"6. Retreating...")
    planner.move_to_pose_with_RRTConnect(pre_place_pose)
    
    print(f"7. Stack task completed.")
    planner.close()
    print("====================================================\n")


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



    # 注意这里改了！！！！！！！！！！！！记得删注释
    # 【新增】魔法操作 1：防滑处理
    mat = red_cube.get_physical_material()
    mat.static_friction = 50.0  # 默认可能只有 0.5，改成 50 超级防滑
    mat.dynamic_friction = 50.0
    red_cube.set_physical_material(mat)

    # 【新增】魔法操作 2：减轻重量（改重力影响的变种）
    # 让方块像泡沫一样轻，爪子稍微碰到就能带起来
    # 注意：太轻可能会被弹飞，建议先试 0.01 kg
    red_cube.set_mass(0.01)



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