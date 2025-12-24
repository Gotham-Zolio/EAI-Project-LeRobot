"""
Reward Function Definitions for RL Training

为不同任务定义奖励函数，用于强化学习训练。
好的奖励函数设计是RL成功的关键。
"""

import numpy as np
from typing import Dict, Any


class RewardFunction:
    """Base reward function class"""
    
    def __init__(self, task: str):
        self.task = task
        self.prev_ee_to_block_dist = None
        self.prev_block_height = None
    
    def compute(self, obs: Dict[str, Any], action: np.ndarray, info: Dict[str, Any]) -> float:
        """Compute reward for current transition"""
        raise NotImplementedError
    
    def reset(self):
        """Reset internal state"""
        self.prev_ee_to_block_dist = None
        self.prev_block_height = None


class LiftReward(RewardFunction):
    """Reward function for lift task
    
    奖励组成：
    1. 靠近物块：-distance_to_block
    2. 抓取接触：contact_bonus
    3. 提升高度：height_reward
    4. 成功完成：success_bonus
    """
    
    def __init__(self):
        super().__init__("lift")
        # Reward weights
        self.w_distance = 1.0          # 距离奖励权重
        self.w_height = 5.0            # 高度奖励权重
        self.w_contact = 2.0           # 接触奖励权重
        self.w_success = 10.0          # 成功奖励权重
        
        # Thresholds
        self.contact_threshold = 0.02  # 2cm内算接触
        self.lift_threshold = 0.10     # 抬起10cm算成功
        self.table_z = 0.015           # 桌面高度
    
    def compute(self, obs: Dict[str, Any], action: np.ndarray, info: Dict[str, Any]) -> float:
        """Compute lift task reward"""
        reward = 0.0
        
        # Extract relevant info (假设环境提供这些信息)
        # 实际使用时需要根据你的环境适配
        ee_pos = info.get("ee_pos", np.zeros(3))          # 末端执行器位置
        block_pos = info.get("block_pos", np.zeros(3))    # 物块位置
        gripper_pos = obs["qpos"][-1] if "qpos" in obs else 1.0  # 夹爪开合度
        
        # 1. Distance reward (靠近物块)
        dist_to_block = np.linalg.norm(ee_pos[:2] - block_pos[:2])  # XY平面距离
        reward_distance = -self.w_distance * dist_to_block
        reward += reward_distance
        
        # 2. Contact reward (接触奖励)
        if dist_to_block < self.contact_threshold:
            # 靠近物块且夹爪闭合
            if gripper_pos < 0.5:  # 夹爪相对闭合
                reward += self.w_contact
        
        # 3. Height reward (提升高度)
        block_height = block_pos[2] - self.table_z
        if block_height > 0.02:  # 物块离开桌面
            reward_height = self.w_height * block_height
            reward += reward_height
            
            # Shaping: 如果持续提升，给予额外奖励
            if self.prev_block_height is not None:
                height_delta = block_height - self.prev_block_height
                if height_delta > 0:
                    reward += self.w_height * height_delta * 2  # 额外提升奖励
        
        # 4. Success reward (任务成功)
        if block_height > self.lift_threshold:
            reward += self.w_success
            # 额外奖励：稳定保持
            if gripper_pos < 0.3:  # 夹紧状态
                reward += self.w_success * 0.5
        
        # Update history
        self.prev_ee_to_block_dist = dist_to_block
        self.prev_block_height = block_height
        
        return reward


class SortReward(RewardFunction):
    """Reward function for sort task"""
    
    def __init__(self):
        super().__init__("sort")
        self.w_pick = 3.0
        self.w_place = 5.0
        self.w_correct_bin = 10.0
        self.w_success = 20.0
    
    def compute(self, obs: Dict[str, Any], action: np.ndarray, info: Dict[str, Any]) -> float:
        reward = 0.0
        
        # Pick phase reward
        ee_pos = info.get("ee_pos", np.zeros(3))
        block_pos = info.get("block_pos", np.zeros(3))
        block_color = info.get("block_color", "unknown")
        
        dist_to_block = np.linalg.norm(ee_pos - block_pos)
        reward += -self.w_pick * dist_to_block
        
        # Place phase reward
        block_height = block_pos[2]
        if block_height > 0.1:  # Block lifted
            # Check if moving toward correct bin
            target_bin_pos = info.get(f"{block_color}_bin_pos", np.zeros(3))
            dist_to_bin = np.linalg.norm(block_pos[:2] - target_bin_pos[:2])
            reward += -self.w_place * dist_to_bin
            
            # Bonus for being near correct bin
            if dist_to_bin < 0.05:
                reward += self.w_correct_bin
        
        # Success: block in correct bin
        if info.get("in_correct_bin", False):
            reward += self.w_success
        
        return reward


class StackReward(RewardFunction):
    """Reward function for stack task"""
    
    def __init__(self):
        super().__init__("stack")
        self.w_approach = 2.0
        self.w_grasp = 3.0
        self.w_lift = 4.0
        self.w_align = 5.0
        self.w_stack = 10.0
        self.w_success = 20.0
    
    def compute(self, obs: Dict[str, Any], action: np.ndarray, info: Dict[str, Any]) -> float:
        reward = 0.0
        
        ee_pos = info.get("ee_pos", np.zeros(3))
        top_block_pos = info.get("top_block_pos", np.zeros(3))
        bottom_block_pos = info.get("bottom_block_pos", np.zeros(3))
        
        # Phase 1: Approach top block
        dist_to_top = np.linalg.norm(ee_pos - top_block_pos)
        reward += -self.w_approach * dist_to_top
        
        # Phase 2: Grasp and lift
        if info.get("grasping_top", False):
            reward += self.w_grasp
            
            top_height = top_block_pos[2]
            if top_height > 0.05:
                reward += self.w_lift * top_height
        
        # Phase 3: Align over bottom block
        if info.get("holding_top", False):
            xy_offset = np.linalg.norm(top_block_pos[:2] - bottom_block_pos[:2])
            reward += -self.w_align * xy_offset
            
            # Bonus for good alignment
            if xy_offset < 0.02:  # Well aligned
                reward += self.w_align * 2
        
        # Phase 4: Stack successfully
        if info.get("stacked", False):
            reward += self.w_stack
            
            # Extra bonus for stable stack
            if info.get("stable_stack", False):
                reward += self.w_success
        
        return reward


def get_reward_function(task: str) -> RewardFunction:
    """Factory function to get reward function for a task"""
    if task == "lift":
        return LiftReward()
    elif task == "sort":
        return SortReward()
    elif task == "stack":
        return StackReward()
    else:
        raise ValueError(f"Unknown task: {task}")


# ========================================
# 使用示例
# ========================================
if __name__ == "__main__":
    # 创建lift任务的奖励函数
    reward_fn = get_reward_function("lift")
    
    # 模拟一个transition
    obs = {"qpos": np.zeros(7)}
    action = np.zeros(7)
    info = {
        "ee_pos": np.array([0.45, 0.25, 0.10]),
        "block_pos": np.array([0.46, 0.24, 0.015]),
    }
    
    reward = reward_fn.compute(obs, action, info)
    print(f"Computed reward: {reward:.4f}")
    
    # 不同阶段的奖励示例
    print("\n奖励函数设计思路：")
    print("1. 靠近阶段：负距离奖励，鼓励接近物块")
    print("2. 接触阶段：接触bonus，鼓励夹爪接触物块")
    print("3. 抓取阶段：夹爪闭合奖励")
    print("4. 提升阶段：高度奖励，与提升高度成正比")
    print("5. 成功阶段：大bonus，完成任务目标")
    print("\n这种分阶段、shaping的奖励设计可以引导RL快速学习")
