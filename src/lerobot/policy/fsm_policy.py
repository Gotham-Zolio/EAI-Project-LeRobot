"""
FSM+IK专家策略（完整版）

核心设计：
1. 纯粹的有限状态机（FSM）：每个任务分解为离散阶段
2. 逆运动学（IK）：多策略鲁棒求解
3. 状态转换：基于时间步或事件触发的确定性转换
4. 调试与诊断：完整的日志和工作空间分析

特性：
- 保留原scripted.py的所有鲁棒性特性
- 多策略IK求解（5种初始猜测）
- 平滑度评分选择最优IK解
- 完整的工作空间诊断
- 详细的调试日志输出
"""

import numpy as np
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

from lerobot.common.kinematics import SimpleKinematics

# 全局调试开关
POLICY_DEBUG = False

# =====================
# 常量定义
# =====================
CM = 0.01

# 夹爪状态
GRIPPER_OPEN = 1.6
GRIPPER_CLOSE = 0.0

# 场景参数
TABLE_Z = 0.0
BLOCK_HALF = 0.015  # 方块半高

# 运动高度（相对于方块顶部）
APPROACH_Z = 0.05   # 接近高度
DESCEND_Z = 0.01    # 下降到抓取位置
RETREAT_Z = 0.25    # 抬升高度

HOLD_STEPS = 15

# =====================
# 基础策略类
# =====================

class BaseFSMPolicy:
    """FSM专家策略基类 - 完整版
    
    包含原scripted.py的所有鲁棒性特性：
    - 多策略IK求解（5种初始猜测）
    - 平滑度评分选择最优解
    - 完整的工作空间诊断
    - 详细的调试日志
    """
    
    def __init__(self, env):
        self.env = env
        self.left_arm = env.left_arm
        self.right_arm = env.right_arm
        self.left_dof = self.left_arm.dof
        self.right_dof = self.right_arm.dof
        
        # 初始化运动学求解器
        self.left_kin = SimpleKinematics(self.left_arm, "gripper_frame_link")
        self.right_kin = SimpleKinematics(self.right_arm, "gripper_frame_link")
        
        # 状态机变量
        self.phase = 0
        self.phase_step = 0
        self.phase_durations = []  # 子类定义
        
        # 随机化参数（增加数据多样性）
        self.grasp_offset = 0.0
        self.grasp_offset_l = 0.0
        self.grasp_offset_r = 0.0
        
        # 调试开关
        self.verbose = False
    
    def reset(self):
        """重置状态机"""
        self.phase = 0
        self.phase_step = 0
    
    def _zero_action(self):
        """返回零动作"""
        return np.zeros(self.left_dof + self.right_dof, dtype=np.float32)
    
    def _current_qpos_from_obs(self, obs):
        """从观测中提取当前关节角度"""
        qpos = obs["qpos"]
        left_qpos = qpos[:self.left_dof]
        right_qpos = qpos[self.left_dof:self.left_dof + self.right_dof]
        return left_qpos, right_qpos
    
    def _safe_compute_ik(self, kin, target_pose, current_qpos, verbose=False):
        """鲁棒IK求解 - 多策略实现（来自原scripted.py）
        
        策略：
        1. 从当前位置开始（保证连续性）
        2. 翻转wrist_flex（通常能找到替代解）
        3. 翻转wrist_roll（进一步增加可能性）
        4. 标准准备姿态
        5. 零位配置
        
        返回最平滑的IK解，或失败时保持当前位置
        """
        # 策略4：翻转wrist_flex（索引3）
        flipped_qpos = current_qpos.copy()
        if len(flipped_qpos) > 3:
            if current_qpos[3] > 0:
                flipped_qpos[3] = current_qpos[3] - np.pi
            else:
                flipped_qpos[3] = current_qpos[3] + np.pi
        
        # 策略：翻转wrist_roll（索引4）
        roll_flipped_qpos = current_qpos.copy()
        if len(roll_flipped_qpos) > 4:
            if current_qpos[4] > 0:
                roll_flipped_qpos[4] = current_qpos[4] - np.pi
            else:
                roll_flipped_qpos[4] = current_qpos[4] + np.pi

        strategies = [
            ("current", current_qpos),
            ("wrist_flex_flipped", flipped_qpos),
            ("wrist_roll_flipped", roll_flipped_qpos),
            ("ready", np.array([0.0, 0.4, 0.8, -0.8, 0.0, current_qpos[-1]], dtype=np.float32)),
            ("zero", np.zeros_like(current_qpos, dtype=np.float32)),
        ]

        # 尝试所有策略，收集成功候选，选择最平滑的
        candidates = []
        for name, init_guess in strategies:
            success, ik = kin.compute_ik(target_pose, initial_qpos=init_guess)

            if verbose and POLICY_DEBUG:
                print(f"[IK] 策略 '{name}': success={success}, result type={type(ik)}")

            if success and isinstance(ik, np.ndarray) and len(ik) == len(current_qpos):
                # 计算平滑度：加权L1距离
                delta = np.abs(ik - current_qpos)
                # wrist关节权重更高，避免大幅反转
                if len(delta) > 3:
                    delta[3] *= 1.5
                if len(delta) > 4:
                    delta[4] *= 1.5
                cost = float(np.sum(delta))
                candidates.append((cost, name, ik))
            elif verbose and POLICY_DEBUG:
                print(f"[IK跳过] 策略 '{name}' 返回无效结果")

        if candidates:
            # 选择成本最小的候选
            candidates.sort(key=lambda x: x[0])
            best_cost, best_name, best_ik = candidates[0]
            if verbose and POLICY_DEBUG:
                print(f"[IK成功] 选择最优策略: '{best_name}' (成本={best_cost:.4f})")
            return best_ik
        
        # 所有策略失败：返回当前位置保持连续性
        if verbose and POLICY_DEBUG:
            print(f"[IK失败] 所有IK策略都失败，保持当前位置")
        return current_qpos.copy()

    def _calculate_grasp_quat(self, arm, target_pos, offset=0.0):
        """计算抓取姿态 - 向下伸展方向（来自原scripted.py）
        
        不是强制完全向下（可能无法到达），而是自然的可达姿态
        """
        # 获取机械臂基座位置
        base_pose = arm.get_pose()
        base_pos = base_pose.p
        
        # 从基座到目标的向量
        to_target = target_pos - base_pos
        distance = np.linalg.norm(to_target)
        
        if distance < 1e-6:
            # 目标在基座：使用默认方向
            yaw = offset
            pitch = -np.pi / 4
            roll = 0.0
        else:
            # 计算偏航角（绕Z轴）
            yaw = np.arctan2(to_target[1], to_target[0]) + offset
            
            # 计算俯仰角（向下倾斜）
            horizontal_dist = np.sqrt(to_target[0]**2 + to_target[1]**2)
            vertical_dist = target_pos[2] - base_pos[2]
            
            if horizontal_dist > 1e-6:
                pitch = -np.arctan2(vertical_dist, horizontal_dist)
                # 限制在合理范围：100°到30°向下
                pitch = np.clip(pitch, -np.pi/1.8, -np.pi/6)
            else:
                pitch = -np.pi / 2.2  # 几乎垂直
            
            roll = 0.0
        
        # 欧拉角转四元数（ZYX顺序）
        rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
        quat_xyzw = rot.as_quat()
        
        # 转换为SAPIEN格式 (w, x, y, z)
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    
    def advance_phase(self):
        """推进到下一阶段"""
        if self.phase_step >= self.phase_durations[self.phase]:
            if self.phase < len(self.phase_durations) - 1:
                self.phase += 1
                self.phase_step = 0
                if self.verbose:
                    print(f"[FSM] 进入阶段 {self.phase}")
        self.phase_step += 1
    
    def get_action(self, obs):
        """生成动作（子类实现）"""
        raise NotImplementedError


# =====================
# Lift任务策略
# =====================

class LiftPolicy(BaseFSMPolicy):
    """Lift任务：抓取并抬升方块（完整版 - 来自scripted.py）
    
    状态机：
    0. XY对齐：移动到方块上方
    1. Z接近：降低到接近高度
    2. 下降：精确下降到抓取位置
    3. 抓取：闭合夹爪
    4. 抬升：向上移动
    5. 保持：维持抬升状态
    
    特性：
    - 事件驱动转换（抓取→抬升需检测夹爪闭合）
    - 位置缓存保证垂直抬升
    - 完整的工作空间诊断
    - 详细的调试日志
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert env.task == "lift", "任务必须是lift"
        
        self.block = env.task_actors[0]
        self.phase_durations = [30, 30, 30, 40, 30, 40]
        
        # 夹爪闭合检测（用于事件驱动转换）
        self.gripper_closed_steps = 0
        self.gripper_closure_threshold = 5
        self.gripper_closure_tolerance = 0.03
        
        # 缓存变量（用于保持稳定的抓取位置）
        self._grasp_xy = None
        self._grasp_block_z = None
        self._grasp_target_quat = None
        self._descend_lock_qpos = None
    
    def reset(self):
        """重置状态机"""
        super().reset()
        self.grasp_offset = np.random.uniform(-0.15, 0.15)
        self.gripper_closed_steps = 0
        self._grasp_xy = None
        self._grasp_block_z = None
        self._grasp_target_quat = None
        self._descend_lock_qpos = None
    
    def _print_workspace_info(self):
        """打印工作空间诊断信息（来自原scripted.py）"""
        arm_base = self.right_arm.get_pose().p
        ee_link = self.right_arm.find_link_by_name("gripper_frame_link")
        ee_pos = ee_link.get_pose().p
        block_pos = self.block.get_pose().p
        block_top_z = block_pos[2] + BLOCK_HALF
        
        print("\n" + "="*80)
        print("工作空间诊断")
        print("="*80)
        print(f"右臂基座: [{arm_base[0]:.4f}, {arm_base[1]:.4f}, {arm_base[2]:.4f}]")
        print(f"夹爪(EE): [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
        print(f"方块中心: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
        print(f"方块顶部Z: {block_top_z:.4f}")
        
        to_block = np.array(block_pos) - np.array(arm_base)
        dist_base_to_block = np.linalg.norm(to_block)
        horizontal_dist = np.linalg.norm(to_block[:2])
        print(f"\n从臂基座到方块的距离:")
        print(f"  总距离:  {dist_base_to_block:.4f} m")
        print(f"  水平:    {horizontal_dist:.4f} m")
        print(f"  竖直:    {to_block[2]:.4f} m")
        
        offset = np.array(ee_pos) - np.array([block_pos[0], block_pos[1], block_top_z])
        offset_norm = np.linalg.norm(offset)
        print(f"\n初始夹爪-方块偏移:")
        print(f"  向量: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
        print(f"  范数: {offset_norm:.4f} m ({offset_norm*100:.1f} cm)")
        
        arm_reach_max = 0.70
        arm_reach_min = 0.15
        if horizontal_dist > arm_reach_max:
            print(f"\n⚠️  警告: 方块可能超出最大到达范围 ({arm_reach_max:.2f}m)!")
        elif horizontal_dist < arm_reach_min:
            print(f"\n⚠️  警告: 方块可能太近 (最小 {arm_reach_min:.2f}m)!")
        else:
            print(f"\n✓ 方块在可达工作空间内")
        print("="*80 + "\n")

    def get_action(self, obs):
        left_qpos, right_qpos = self._current_qpos_from_obs(obs)
        
        # =====================
        # 状态机推进（事件驱动for GRASP→LIFT）
        # =====================
        advance_phase = True
        
        # GRASP→LIFT: 严格夹爪闭合检测
        if self.phase == 3:  # 抓取阶段
            gripper_closed = abs(right_qpos[-1] - GRIPPER_CLOSE) < self.gripper_closure_tolerance

            if gripper_closed:
                self.gripper_closed_steps += 1
            else:
                self.gripper_closed_steps = 0

            if not (self.gripper_closed_steps >= self.gripper_closure_threshold):
                advance_phase = False
            
            if self.phase_step % 5 == 0:
                print(f"[LiftPolicy 抓取] step={self.phase_step} 夹爪位置={right_qpos[-1]:.6f} 闭合步数={self.gripper_closed_steps}/{self.gripper_closure_threshold}")
        else:
            self.gripper_closed_steps = 0
        
        # 更新阶段
        if (
            advance_phase
            and self.phase < len(self.phase_durations) - 1
            and self.phase_step >= self.phase_durations[self.phase]
        ):
            self.phase += 1
            self.phase_step = 0
        self.phase_step += 1
        
        # 在回合开始时随机化抓取偏航角
        if self.phase == 0 and self.phase_step == 1:
            self.grasp_offset = np.random.uniform(-0.15, 0.15)
        
        # 获取方块位置
        bx, by, bz = self.block.get_pose().p
        block_top_z = bz + BLOCK_HALF

        # 在GRASP入口缓存位置
        if self.phase == 3 and self.phase_step == 1:
            self._grasp_xy = np.array([bx, by], dtype=np.float32)
            self._grasp_block_z = bz
            grasp_target_pos = np.array([bx, by, bz + BLOCK_HALF], dtype=np.float32)
            self._grasp_target_quat = self._calculate_grasp_quat(self.right_arm, grasp_target_pos, offset=self.grasp_offset)
            
            if self._descend_lock_qpos is None:
                self._descend_lock_qpos = right_qpos.copy()
                if self.verbose and POLICY_DEBUG:
                    print(f"[LiftPolicy] 警告: _descend_lock_qpos未从DESCEND缓存，使用当前qpos")

        # 诊断夹爪-方块偏移
        ee_link = self.right_arm.find_link_by_name("gripper_frame_link")
        ee_pos = np.array(ee_link.get_pose().p, dtype=np.float32)
        block_anchor = np.array([bx, by, block_top_z], dtype=np.float32)
        offset_vec = ee_pos - block_anchor
        offset_norm = float(np.linalg.norm(offset_vec))
        
        # 特殊监控DESCEND阶段：追踪Z轴接触
        if self.phase == 2:
            z_distance = block_top_z - ee_pos[2]
            if self.phase_step % 5 == 0:
                print(f"[LiftPolicy 下降] step={self.phase_step} EE_Z={ee_pos[2]:.4f} 方块顶部Z={block_top_z:.4f} Z距离={z_distance:.6f} XY偏移={np.linalg.norm(offset_vec[:2]):.6f}")
        
        if self.verbose and POLICY_DEBUG:
            print(f"[LiftPolicy] EE-方块偏移 | 阶段={self.phase} 步数={self.phase_step} | 向量={offset_vec} | 范数={offset_norm:.4f} m")
        
        # 根据阶段确定目标位置和夹爪状态
        if self.phase == 0:
            # XY对齐
            current_ee_z = ee_link.get_pose().p[2]
            safe_high_z = TABLE_Z + 0.17
            target_pos = np.array([bx, by, safe_high_z], dtype=np.float32)
            gripper = GRIPPER_OPEN
            phase_name = "XY对齐"
        elif self.phase == 1:
            # Z接近
            target_pos = np.array([bx, by, block_top_z + APPROACH_Z], dtype=np.float32)
            gripper = GRIPPER_OPEN
            phase_name = "Z接近"
        elif self.phase == 2:
            # 下降
            target_pos = np.array([bx, by, block_top_z + DESCEND_Z], dtype=np.float32)
            gripper = GRIPPER_OPEN
            phase_name = "下降"
        elif self.phase == 3:
            # 抓取
            grasp_xy = self._grasp_xy if self._grasp_xy is not None else np.array([bx, by], dtype=np.float32)
            grasp_block_z = self._grasp_block_z if self._grasp_block_z is not None else bz
            target_pos = np.array([grasp_xy[0], grasp_xy[1], grasp_block_z + BLOCK_HALF], dtype=np.float32)
            gripper = GRIPPER_CLOSE
            phase_name = "抓取"
        elif self.phase == 4:
            # 抬升
            lift_xy = self._grasp_xy if self._grasp_xy is not None else np.array([bx, by], dtype=np.float32)
            lift_block_z = self._grasp_block_z if self._grasp_block_z is not None else bz
            target_pos = np.array([lift_xy[0], lift_xy[1], lift_block_z + BLOCK_HALF + RETREAT_Z], dtype=np.float32)
            gripper = GRIPPER_CLOSE
            phase_name = "抬升"
        else:
            # 保持
            hold_xy = self._grasp_xy if self._grasp_xy is not None else np.array([bx, by], dtype=np.float32)
            hold_block_z = self._grasp_block_z if self._grasp_block_z is not None else bz
            target_pos = np.array([hold_xy[0], hold_xy[1], hold_block_z + BLOCK_HALF + RETREAT_Z], dtype=np.float32)
            gripper = GRIPPER_CLOSE
            phase_name = "保持"

        # 暴露目标信息供外部使用
        try:
            self.last_phase_name = phase_name
            self.last_target_pos = target_pos.copy()
        except Exception:
            pass

        # 在阶段入口输出状态
        if self.phase_step == 1:
            print(f"[LiftPolicy] 状态: {phase_name} (阶段={self.phase})")
        
        # 计算目标姿态和IK解
        if self.phase == 2:  # DESCEND: 每步计算新IK以逐步下降
            quat = self._calculate_grasp_quat(self.right_arm, target_pos, offset=self.grasp_offset)
            target_pose = Pose(p=target_pos, q=quat)
            target_right_qpos = self._safe_compute_ik(self.right_kin, target_pose, right_qpos, verbose=self.verbose)
            self._descend_lock_qpos = target_right_qpos.copy()
        elif self.phase == 3:  # GRASP: 锁定关节角度
            quat = self._grasp_target_quat.copy() if self._grasp_target_quat is not None else self._calculate_grasp_quat(self.right_arm, target_pos, offset=self.grasp_offset)
            if self._descend_lock_qpos is not None:
                target_right_qpos = self._descend_lock_qpos.copy()
            else:
                target_right_qpos = right_qpos.copy()
        else:  # 其他阶段: 计算新IK
            quat = self._calculate_grasp_quat(self.right_arm, target_pos, offset=self.grasp_offset)
            target_pose = Pose(p=target_pos, q=quat)
            target_right_qpos = self._safe_compute_ik(self.right_kin, target_pose, right_qpos, verbose=self.verbose)
            
            # 调试输出LIFT阶段
            if self.phase == 4 and self.phase_step == 1:
                current_ee_pos = ee_link.get_pose().p
                print(f"\n[LiftPolicy 抬升] 阶段入口诊断:")
                print(f"  目标EE位置:     {target_pos}")
                print(f"  当前EE位置:     {current_ee_pos}")
                print(f"  位置差:         {target_pos - current_ee_pos}")
                print(f"  位置差范数:     {np.linalg.norm(target_pos - current_ee_pos):.6f} m")
                print(f"  当前qpos:       {right_qpos}")
                print(f"  IK解qpos:       {target_right_qpos}")
                print(f"  关节差:         {target_right_qpos - right_qpos}")
                print(f"  缓存XY:         {self._grasp_xy}")
                print(f"  缓存block_z:    {self._grasp_block_z}")
                print()
        
        # 调试输出阶段入口
        if self.phase_step == 1 and self.verbose and POLICY_DEBUG:
            arm_base = self.right_arm.get_pose().p
            to_target = target_pos - np.array(arm_base)
            target_dist = np.linalg.norm(to_target)
            target_horizontal = np.linalg.norm(to_target[:2])
            print(f"\n[阶段 {self.phase} {phase_name}]")
            print(f"  方块位置:    [{bx:.4f}, {by:.4f}, {bz:.4f}]")
            print(f"  方块顶部Z:   {block_top_z:.4f}")
            print(f"  目标EE:      [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            print(f"  目标四元数:  [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            print(f"  从臂基座到目标的距离:")
            print(f"    总距离:    {target_dist:.4f} m")
            print(f"    水平:      {target_horizontal:.4f} m")
            print(f"  当前qpos:    {right_qpos}")
            print(f"  目标qpos:    {target_right_qpos}")
            print(f"  实际EE:      {ee_link.get_pose().p}")
        
        target_right_qpos[-1] = gripper
        
        # 在XY对齐阶段应用wrist_flex翻转
        if self.phase == 0 and len(target_right_qpos) > 3:
            target_right_qpos[3] = -target_right_qpos[3]
            if self.verbose and POLICY_DEBUG:
                print(f"[LiftPolicy] flip_wrist_flex应用 (XY对齐): wrist_flex -> {target_right_qpos[3]:.4f}")
        
        # 构造动作：左臂保持不动，右臂跟随IK
        action = self._zero_action()
        action[:self.left_dof] = left_qpos
        action[self.left_dof:self.left_dof + self.right_dof] = target_right_qpos
        
        return action


# =====================
# Stack任务策略
# =====================

class StackPolicy(BaseFSMPolicy):
    """Stack任务：将红色方块叠到绿色方块上
    
    状态机：
    0. 接近红块：移动到红色方块上方
    1. 下降：降低到抓取位置
    2. 抓取：闭合夹爪抓住红块
    3. 抬升：抬起红色方块
    4. 移动：移动到绿色方块上方
    5. 释放：张开夹爪释放
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert env.task == "stack", "任务必须是stack"
        
        self.red, self.green = env.task_actors
        self.phase_durations = [40, 40, 30, 40, 40, 30]
    
    def reset(self):
        super().reset()
        self.grasp_offset = np.random.uniform(-0.15, 0.15)

    def get_action(self, obs):
        left_qpos, right_qpos = self._current_qpos_from_obs(obs)
        
        if self.phase == 0 and self.phase_step == 1:
            self.grasp_offset = np.random.uniform(-0.15, 0.15)
        
        red_pos = self.red.get_pose().p
        green_pos = self.green.get_pose().p
        
        # 确定目标位置和夹爪状态
        if self.phase == 0:
            target_pos = np.array([red_pos[0], red_pos[1], red_pos[2] + APPROACH_Z])
            gripper = GRIPPER_OPEN
        elif self.phase == 1:
            target_pos = np.array([red_pos[0], red_pos[1], red_pos[2] + DESCEND_Z])
            gripper = GRIPPER_OPEN
        elif self.phase == 2:
            target_pos = np.array([red_pos[0], red_pos[1], red_pos[2] + DESCEND_Z])
            gripper = GRIPPER_CLOSE
        elif self.phase == 3:
            target_pos = np.array([red_pos[0], red_pos[1], red_pos[2] + RETREAT_Z])
            gripper = GRIPPER_CLOSE
        elif self.phase == 4:
            target_pos = np.array([green_pos[0], green_pos[1], TABLE_Z + BLOCK_HALF * 3])
            gripper = GRIPPER_CLOSE
        else:
            target_pos = np.array([green_pos[0], green_pos[1], TABLE_Z + BLOCK_HALF * 3])
            gripper = GRIPPER_OPEN
        
        # 计算IK
        target_quat = self._calculate_grasp_quat(self.right_arm, target_pos, self.grasp_offset)
        target_pose = Pose(p=target_pos, q=target_quat)
        target_right_qpos = self._safe_compute_ik(self.right_kin, target_pose, right_qpos, verbose=self.verbose)
        target_right_qpos[-1] = gripper
        
        # 推进状态机
        self.advance_phase()
        
        # 构造动作
        action = self._zero_action()
        action[:self.left_dof] = left_qpos
        action[self.left_dof:] = target_right_qpos
        
        return action


# =====================
# Sort任务策略
# =====================

class SortPolicy(BaseFSMPolicy):
    """Sort任务：双臂分拣
    
    - 左臂：绿色方块 → 左侧容器
    - 右臂：红色方块 → 右侧容器
    
    状态机：
    0. 左臂抓取并放置绿色方块
    1. 右臂抓取并放置红色方块
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert env.task == "sort", "任务必须是sort"
        
        # 识别红绿方块
        a0, a1 = env.task_actors
        self.red, self.green = (a0, a1) if "Red" in a0.name else (a1, a0)
        
        # 目标容器位置
        y = 0.25
        self.left_bin = np.array([0.12, y, TABLE_Z + BLOCK_HALF])
        self.right_bin = np.array([0.48, y, TABLE_Z + BLOCK_HALF])
        
        self.phase_durations = [80, 80]
    
    def reset(self):
        super().reset()
        self.grasp_offset_l = np.random.uniform(-0.15, 0.15)
        self.grasp_offset_r = np.random.uniform(-0.15, 0.15)
    
    def get_action(self, obs):
        left_qpos, right_qpos = self._current_qpos_from_obs(obs)
        
        # 初始化偏移
        if self.phase == 0 and self.phase_step == 1:
            self.grasp_offset_l = np.random.uniform(-0.15, 0.15)
        if self.phase == 1 and self.phase_step == 1:
            self.grasp_offset_r = np.random.uniform(-0.15, 0.15)
        
        green_pos = self.green.get_pose().p
        red_pos = self.red.get_pose().p
        
        # 阶段0：左臂处理绿色方块
        if self.phase == 0:
            steps = self.phase_durations[0]
            if self.phase_step < steps // 2:
                # 前半段：抓取
                if self.phase_step < steps // 4:
                    target_pos_l = np.array([green_pos[0], green_pos[1], green_pos[2] + APPROACH_Z])
                    gripper_l = GRIPPER_OPEN
                else:
                    target_pos_l = np.array([green_pos[0], green_pos[1], green_pos[2] + DESCEND_Z])
                    gripper_l = GRIPPER_CLOSE
            else:
                # 后半段：移动并释放
                target_pos_l = self.left_bin.copy()
                gripper_l = GRIPPER_CLOSE if self.phase_step < steps * 3 / 4 else GRIPPER_OPEN
            
            quat_l = self._calculate_grasp_quat(self.left_arm, target_pos_l, self.grasp_offset_l)
            pose_l = Pose(p=target_pos_l, q=quat_l)
            target_left_qpos = self._safe_compute_ik(self.left_kin, pose_l, left_qpos, verbose=self.verbose)
            target_left_qpos[-1] = gripper_l
            target_right_qpos = right_qpos
        
        # 阶段1：右臂处理红色方块
        else:
            steps = self.phase_durations[1]
            if self.phase_step < steps // 2:
                # 前半段：抓取
                if self.phase_step < steps // 4:
                    target_pos_r = np.array([red_pos[0], red_pos[1], red_pos[2] + APPROACH_Z])
                    gripper_r = GRIPPER_OPEN
                else:
                    target_pos_r = np.array([red_pos[0], red_pos[1], red_pos[2] + DESCEND_Z])
                    gripper_r = GRIPPER_CLOSE
            else:
                # 后半段：移动并释放
                target_pos_r = self.right_bin.copy()
                gripper_r = GRIPPER_CLOSE if self.phase_step < steps * 3 / 4 else GRIPPER_OPEN
            
            quat_r = self._calculate_grasp_quat(self.right_arm, target_pos_r, self.grasp_offset_r)
            pose_r = Pose(p=target_pos_r, q=quat_r)
            target_right_qpos = self._safe_compute_ik(self.right_kin, pose_r, right_qpos, verbose=self.verbose)
            target_right_qpos[-1] = gripper_r
            target_left_qpos = left_qpos
        
        # 推进状态机
        self.advance_phase()
        
        # 构造动作
        action = self._zero_action()
        action[:self.left_dof] = target_left_qpos
        action[self.left_dof:] = target_right_qpos
        
        return action
