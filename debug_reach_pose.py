#!/usr/bin/env python
"""Debug script to understand why reach_pose becomes huge in collect_data"""
import sys
import numpy as np
import sapien

# Test with different Pose multiplication orders
def test_pose_multiplication():
    # Simulate the grasp pose from debug_poses.py result
    cube_pos = np.array([0.43614864, 0.17679878, 0.015])
    
    # Create reaching offset pose
    reaching_offset = sapien.Pose(p=np.array([0, 0.02, 0.03]))
    
    # Simulate grasp pose (identity rotation + cube position)
    grasp_pose = sapien.Pose(p=cube_pos)
    
    print(f"Reaching offset: p={reaching_offset.p}, q={reaching_offset.q}")
    print(f"Grasp pose: p={grasp_pose.p}, q={grasp_pose.q}")
    
    # Order 1: reaching_offset * grasp_pose
    result1 = reaching_offset * grasp_pose
    print(f"\nreaching_offset * grasp_pose:")
    print(f"  p={result1.p}")
    print(f"  q={result1.q}")
    
    # Order 2: grasp_pose * reaching_offset  
    result2 = grasp_pose * reaching_offset
    print(f"\ngrasp_pose * reaching_offset:")
    print(f"  p={result2.p}")
    print(f"  q={result2.q}")
    
    # What should happen: we want the reaching_offset to be relative to the grasp_pose
    # So we should multiply: grasp_pose * reaching_offset
    # Actually, let's think: if grasp_pose is at [0.436, 0.177, 0.015] and we want
    # to reach 2cm away in Y direction, the reach position should be [0.436, 0.197, 0.045]
    # which is [0.436, 0.177+0.02, 0.015+0.03]
    
    print(f"\nExpected reach_pose: p=[0.436, 0.197, 0.045]")
    print(f"Got from reaching_offset * grasp_pose: p={result1.p}")
    print(f"Got from grasp_pose * reaching_offset: p={result2.p}")


if __name__ == "__main__":
    test_pose_multiplication()
