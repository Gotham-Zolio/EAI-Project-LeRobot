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
# 使用新的FSM策略（精简版）
from lerobot.policy.fsm_policy import LiftPolicy, StackPolicy, SortPolicy
# 如果需要使用旧版策略，改为：
# from lerobot.policy.scripted import LiftPolicy, StackPolicy, SortPolicy


@dataclass
class CollectionConfig:
    """数据采集配置"""
    task: str = "lift"  # 任务类型: lift, stack, sort
    num_episodes: int = 10  # 采集回合数
    save_dir: str = "data/raw"  # 保存目录
    max_steps: int = 300  # 每回合最大步数
    headless: bool = True  # 无头模式
    verbose: bool = False  # 详细输出
    web_viewer: bool = False  # 是否启用web可视化
    port: int = 5000  # web viewer端口
    sleep_viewer_sec: float = 0.1  # viewer刷新间隔


class FSMDataCollector:
    """基于FSM+IK的数据采集器"""
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.env = None
        self.policy = None
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
            headless=self.config.headless
        )
        
        # 选择对应策略
        policy_map = {
            "lift": LiftPolicy,
            "stack": StackPolicy,
            "sort": SortPolicy
        }
        PolicyClass = policy_map[self.config.task]
        self.policy = PolicyClass(self.env)
        self.policy.verbose = self.config.verbose
        
    def collect_episode(self, episode_id: int):
        """采集单个回合"""
        # 重置环境和策略
        obs, _ = self.env.reset()
        self.policy.reset()  # 使用策略的reset方法
        
        # 存储轨迹
        trajectory = {
            "qpos": [],
            "action": [],
            "reward": [],
            "done": [],
            "images": {cam: [] for cam in self.cameras}
        }
        
        done = False
        step = 0
        
        print(f"\n回合 {episode_id + 1}/{self.config.num_episodes}")
        # WebViewer状态更新
        if self.viewer_app:
            self.viewer_app.update_status(
                mode="Data Collection",
                episode=episode_id + 1,
                total_episodes=self.config.num_episodes,
                task=self.config.task,
            )
        # 运行回合
        while not done and step < self.config.max_steps:
            step += 1
            # WebViewer帧推送（推送所有可用的摄像头，即使不用于采集）
            if self.viewer_app:
                # 获取所有可用的摄像头帧
                all_available_cams = list(obs["images"].keys())
                frames = {cam: obs["images"][cam] for cam in all_available_cams if cam in obs["images"]}
                if step == 1:
                    print(f"  Available cameras in env: {all_available_cams}")
                    print(f"  Sending frames to viewer: {list(frames.keys())}")
                    for cam, frame in frames.items():
                        print(f"    {cam}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
                self.viewer_app.update_frames(frames)
                import time as _time; _time.sleep(self.config.sleep_viewer_sec)
            # FSM策略生成动作
            action = self.policy.get_action(obs)
            # 执行动作
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            # 记录数据
            trajectory["qpos"].append(obs["qpos"])
            trajectory["action"].append(action)
            trajectory["reward"].append(reward)
            trajectory["done"].append(done)
            # 记录图像
            for cam in self.cameras:
                if cam in obs["images"]:
                    trajectory["images"][cam].append(obs["images"][cam])
            obs = next_obs
            # 每50步输出进度
            if step % 50 == 0:
                print(f"  步数: {step}, 阶段: {self.policy.phase}")
        
        # 添加最终状态图像
        for cam in self.cameras:
            if cam in obs["images"]:
                trajectory["images"][cam].append(obs["images"][cam])
        
        success = any(trajectory["reward"])
        print(f"回合完成: 步数={step}, 成功={success}")
        
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
