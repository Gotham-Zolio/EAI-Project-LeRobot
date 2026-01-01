#!/usr/bin/env python3
"""Debug script to check pose calculations."""

import sys
sys.path.insert(0, "src")

import numpy as np
import sapien
from lerobot.envs.gym_env import LeRobotGymEnv
from lerobot.common.motion_planning.base_motionplanner.utils import get_actor_obb, compute_grasp_info_by_obb

# Create environment
env = LeRobotGymEnv(task="lift", headless=True, max_steps=300)
actual_env = env.env

print(f"\n=== Environment Setup ===")
print(f"Cube position: {actual_env.cube.pose}")
print(f"Cube position (sp): {actual_env.cube.pose.sp}")
print(f"Cube position (sp.p): {actual_env.cube.pose.sp.p}")

# Get OBB
cube = actual_env.cube  # Use .cube directly from SO101TaskEnv
print(f"\nCube actor: {cube}")
obb = get_actor_obb(cube)
print(f"OBB center: {obb.centroid}")
print(f"OBB primitive transform:\n{obb.primitive.transform}")

# Compute grasp info
approaching = np.array([0, 0, -1])
agent = actual_env.agent_right
tcp_pose = agent.tcp_pose.sp
print(f"\nTCP pose: {tcp_pose}")
tcp_T = tcp_pose.to_transformation_matrix()
target_closing = tcp_T[:3, 1]
print(f"Target closing: {target_closing}")

grasp_info = compute_grasp_info_by_obb(
    obb,
    approaching=approaching,
    target_closing=target_closing,
    depth=0.025,
)

print(f"\nGrasp info:")
print(f"  approaching: {grasp_info['approaching']}")
print(f"  closing: {grasp_info['closing']}")
print(f"  center: {grasp_info['center']}")
print(f"  extents: {grasp_info['extents']}")

# Build grasp pose
grasp_pose = agent.build_grasp_pose(
    grasp_info['approaching'],
    grasp_info['closing'],
    cube.pose.sp.p
)
print(f"\nGrasp pose (from agent.build_grasp_pose):")
print(f"  p: {grasp_pose.p}")
print(f"  q: {grasp_pose.q}")

# Apply rotation transform
grasp_pose_rot = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
print(f"\nGrasp pose after rotation:")
print(f"  p: {grasp_pose_rot.p}")
print(f"  q: {grasp_pose_rot.q}")

# Reach pose
reach_pose = sapien.Pose([0, 0.02, 0.03]) * grasp_pose_rot
print(f"\nReach pose:")
print(f"  p: {reach_pose.p}")
print(f"  q: {reach_pose.q}")

env.close()
