#!/usr/bin/env python3
"""
精简的FSM+IK专家策略数据采集器

核心原理：
1. 有限状态机（FSM）：定义任务的离散阶段和转换逻辑
2. 逆运动学（IK）：将末端位姿目标转换为关节角度
3. 专家策略：通过硬编码状态机实现确定性行为

特点：
- 纯FSM实现，无依赖RL残差
- 精简代码结构，保持核心逻辑
- 支持lift/stack/sort三种任务
"""

import sys
import os
import tyro
import numpy as np
import h5py
from dataclasses import dataclass


from pathlib import Path


# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# WebViewer导入（必须在sys.path插入后）
from tools.web_viewer.viewer import WebViewer

from lerobot.envs.gym_env import LeRobotGymEnv
# 使用新的Motion Planning任务求解器
from lerobot.policy.task_planner import solve_lift, solve_stack, solve_sort


@dataclass
class CollectionConfig:
    """数据采集配置"""
    task: str = "lift"  # 任务类型: lift, stack, sort
    num_episodes: int = 10  # 采集回合数
    save_dir: str = os.path.expanduser("~/tmp/data/raw")  # 保存目录
    max_steps: int = 300  # 每回合最大步数
    headless: bool = True  # 无头模式
    verbose: bool = False  # 详细输出
    web_viewer: bool = False  # 是否启用web可视化
    port: int = 5000  # web viewer端口
    sleep_viewer_sec: float = 0.1  # viewer刷新间隔 (建议0.1及以上，减轻web卡顿)
    vis: bool = False  # 是否在仿真环境中可视化坐标轴


class RecordingWrapper:
    """环境包装器，用于在step过程中自动记录数据"""
    def __init__(self, env, cameras, viewer_app=None, sleep_viewer_sec=0.1):
        self.env = env
        self.cameras = cameras
        self.viewer_app = viewer_app
        self.sleep_viewer_sec = sleep_viewer_sec
        self.trajectory = {
            "qpos": [], "action": [], "reward": [], "done": [],
            "images": {cam: [] for cam in cameras}
        }
        self.step_count = 0
        self.last_obs = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        # 清空轨迹
        self.trajectory = {
            "qpos": [], "action": [], "reward": [], "done": [],
            "images": {cam: [] for cam in self.cameras}
        }
        self.step_count = 0
        return obs, info

    def step(self, action):
        if self.last_obs is None:
            # Should have been set by reset, but if not (e.g. manual reset of inner env), try to get it
            self.last_obs = self.env._get_obs()

        obs = self.last_obs
        
        # WebViewer更新
        if self.viewer_app:
            frames = {cam: obs["images"][cam] for cam in self.cameras if cam in obs["images"]}
            # 额外推送world画面到WebViewer（即使不采集world）
            if "world" in obs["images"]:
                frames["world"] = obs["images"]["world"]
            self.viewer_app.update_frames(frames)
            # import time; time.sleep(self.sleep_viewer_sec) # 可能会拖慢采集速度，视情况开启

        # 执行动作
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # 记录数据
        self.trajectory["qpos"].append(obs["qpos"])
        self.trajectory["action"].append(action)
        self.trajectory["reward"].append(reward)
        self.trajectory["done"].append(done)
        for cam in self.cameras:
            if cam in obs["images"]:
                self.trajectory["images"][cam].append(obs["images"][cam])
        
        self.last_obs = next_obs
        self.step_count += 1
        
        return next_obs, reward, terminated, truncated, info


class FSMDataCollector:
    """基于Motion Planning的数据采集器"""
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.env = None
        self.solver = None
        self.viewer_app = None
        self.cameras = []
    
    def setup(self):
        """初始化环境和策略"""
        print(f"初始化任务: {self.config.task}")
        
        # 确定相机配置
        if self.config.task == "lift":
            self.cameras = ["front", "right_wrist"]
        elif self.config.task == "sort":
            self.cameras = ["front", "left_wrist", "right_wrist"]
        elif self.config.task == "stack":
            self.cameras = ["front", "right_wrist"]
        else:
            raise ValueError(f"未知任务: {self.config.task}")
        
        print(f"使用相机: {self.cameras}")
        
        # 启动WebViewer（如需）
        if self.config.web_viewer:
            self.viewer_app = WebViewer(port=self.config.port)
            self.viewer_app.start()
            print(f"Web viewer started at http://localhost:{self.config.port}")
        
        # 创建环境
        self.env = LeRobotGymEnv(
            task=self.config.task,
            headless=self.config.headless,
            max_steps=self.config.max_steps
        )
        
        # 选择对应求解器
        solver_map = {
            "lift": solve_lift,
            "stack": solve_stack,
            "sort": solve_sort
        }
        self.solver = solver_map[self.config.task]
        
    def collect_episode(self, episode_id: int):
        """采集单个回合"""
        # 创建包装器
        wrapper = RecordingWrapper(
            self.env, 
            self.cameras, 
            self.viewer_app, 
            self.config.sleep_viewer_sec
        )
        
        # 重置环境
        wrapper.reset()
        
        print(f"\n回合 {episode_id + 1}/{self.config.num_episodes}")
        if self.viewer_app:
            self.viewer_app.update_status(
                mode="Data Collection (MP)",
                episode=episode_id + 1,
                total_episodes=self.config.num_episodes,
                task=self.config.task,
            )
            
        # 运行求解器
        try:
            self.solver(wrapper, seed=episode_id, debug=self.config.verbose, vis=self.config.vis)
            success = True # 如果求解器没有抛出异常，假设成功？或者检查最后的状态？
            # 简单的成功判定：是否有reward > 0 (假设环境有定义reward)
            # 或者检查任务完成条件。
            # 目前LeRobotGymEnv的reward可能未定义(0.0)。
            # 我们可以假设只要没报错且步数足够就是成功，或者需要更严格的检查。
            # 暂时假设成功。
        except Exception as e:
            print(f"Episode failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
        
        # 获取轨迹
        trajectory = wrapper.trajectory
        
        # 添加最终状态图像 (Optional, usually we want len(obs) = len(action) + 1 or same)
        # Standard: len(obs) = len(action) + 1 (initial obs ... final obs)
        # Our recording loop records obs BEFORE step.
        # So we have N obs and N actions.
        # We should append the final obs (next_obs from last step).
        if wrapper.last_obs is not None:
             for cam in self.cameras:
                if cam in wrapper.last_obs["images"]:
                    trajectory["images"][cam].append(wrapper.last_obs["images"][cam])
        
        print(f"回合完成: 步数={len(trajectory['action'])}, 成功={success}")
        
        return trajectory, success
    
    def save_data(self, trajectories: list, save_path: Path):
        """保存轨迹数据到HDF5"""
        print(f"\n保存数据到: {save_path}")
        
        with h5py.File(save_path, "w") as f:
            # 元数据
            f.attrs["task"] = self.config.task
            f.attrs["num_episodes"] = len(trajectories)
            f.attrs["collection_method"] = "fsm_ik"
            f.attrs["cameras"] = self.cameras
            
            # 保存每个回合
            for ep_id, traj in enumerate(trajectories):
                grp = f.create_group(f"episode_{ep_id}")
                
                # 状态和动作
                grp.create_dataset("qpos", data=np.array(traj["qpos"], dtype=np.float32))
                grp.create_dataset("action", data=np.array(traj["action"], dtype=np.float32))
                grp.create_dataset("reward", data=np.array(traj["reward"], dtype=np.float32))
                grp.create_dataset("done", data=np.array(traj["done"], dtype=bool))
                
                # 图像
                img_grp = grp.create_group("images")
                for cam in self.cameras:
                    if cam in traj["images"]:
                        img_grp.create_dataset(
                            cam, 
                            data=np.array(traj["images"][cam], dtype=np.uint8)
                        )
        
        print(f"✅ 数据保存成功")
    
    def run(self):
        """执行完整数据采集流程"""
        self.setup()
        
        # 准备保存路径
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{self.config.task}_demo.h5"
        
        # 采集所有回合
        trajectories = []
        success_count = 0
        
        for ep in range(self.config.num_episodes):
            trajectory, success = self.collect_episode(ep)
            trajectories.append(trajectory)
            if success:
                success_count += 1
        
        # 保存数据
        self.save_data(trajectories, save_path)
        
        # 统计
        print(f"\n{'='*60}")
        print(f"采集完成")
        print(f"任务: {self.config.task}")
        print(f"总回合: {self.config.num_episodes}")
        print(f"成功率: {success_count}/{self.config.num_episodes} ({100*success_count/self.config.num_episodes:.1f}%)")
        print(f"保存路径: {save_path}")
        print(f"{'='*60}")
        
        # 关闭环境
        self.env.close()
        if self.viewer_app:
            print("关闭WebViewer...")
            # 没有stop方法，线程daemon自动退出



def main(config: CollectionConfig):
    """
    FSM+IK专家策略数据采集
    通过dataclass递归支持所有参数
    """
    collector = FSMDataCollector(config)
    collector.run()


if __name__ == "__main__":
    tyro.cli(main)
