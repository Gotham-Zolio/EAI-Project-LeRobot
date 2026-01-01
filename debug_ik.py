#!/usr/bin/env python
"""Debug IK reachability"""
import numpy as np
import sapien
from transforms3d import euler

# The target reach_pose from the debug output
reach_pose_p = np.array([0.44559512, 0.19455436, 0.04500001])
reach_pose_q = np.array([0., -0.7071064, -0.7071071, 0.])

# Convert quaternion to Euler angles to understand the orientation
euler_angles = euler.quat2euler(reach_pose_q)
print(f"Target reach_pose position: {reach_pose_p}")
print(f"Target reach_pose quaternion: {reach_pose_q}")
print(f"Target reach_pose Euler (roll, pitch, yaw): {np.degrees(euler_angles)}")

# The rotation seems to be 180 degrees around Y axis
# Let's verify by creating a rotation matrix
from transforms3d.quaternions import quat2mat
rot_matrix = quat2mat(reach_pose_q)
print(f"\nRotation matrix from quaternion:")
print(rot_matrix)

# The initial TCP pose when all joints are at zero
# Let's check what orientation the TCP has at zero configuration
# From debug: grasp_pose before rotation q=[0.0000000e+00 1.0000000e+00 5.4197034e-07 0.0000000e+00]
# This is approximately [0, 1, 0, 0] which is 180 degrees around Y axis (roll=180)
initial_tcp_q = np.array([0., 1., 5.4197034e-07, 0.])
initial_euler = euler.quat2euler(initial_tcp_q)
print(f"\nInitial TCP at cube (before extra rotation):")
print(f"  Quaternion: {initial_tcp_q}")
print(f"  Euler: {np.degrees(initial_euler)}")

# The extra rotation applied: sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
# Which is q=[-0.707, 0, 0, 0.707]
extra_rotation_q = np.array([-1, 0, 0, 1]) / np.sqrt(2)
print(f"\nExtra rotation applied: {extra_rotation_q}")
extra_euler = euler.quat2euler(extra_rotation_q)
print(f"  Euler: {np.degrees(extra_euler)}")

# So the reach_pose orientation should have a different Z or something
# Maybe the issue is that we need a different orientation?
# Let me check what orientation would be more natural for grasping
print("\n\nWhat if we use different orientations?")

# Try identity orientation
identity_q = np.array([1., 0., 0., 0.])
print(f"Identity quaternion: {identity_q}")
print(f"  Euler: {np.degrees(euler.quat2euler(identity_q))}")

# Try the initial TCP orientation (180 around Y)
print(f"\nInitial TCP orientation (180 around Y): {initial_tcp_q}")
print(f"  Euler: {np.degrees(euler.quat2euler(initial_tcp_q))}")
