import numpy as np
import sapien
from lerobot.common.kinematics import SimpleKinematics
from scipy.spatial.transform import Rotation as R

class LiftPolicy:
    def __init__(self, env):
        self.env = env
        # Assuming left arm is used for the task
        # Need to verify the end effector link name from URDF. Usually it's the last link.
        # Let's assume "link6" or similar. We will check this.
        # For SO-100/101, it might be "moving_jaw" or "Link_6"
        # Let's try to find a valid link name dynamically or assume a standard one.
        # Based on previous context, let's assume "Link_6" or similar.
        # Actually, let's look at the robot links in the env if possible, but here we hardcode for now.
        # We will use "Link_5" or "Link_6" as the wrist.
        # Let's assume "Link_5" is the wrist/end-effector base for now.
        self.ee_link_name = "Link_5" 
        
        # We need to initialize kinematics after the robot is loaded
        self.kinematics = None
        
        self.stage = "APPROACH"
        self.gripper_state = 1.0 # Open
        self.target_block_name = "box" # The name prefix for the block actor
        
    def _init_kinematics(self):
        if self.kinematics is None:
            # Find the correct link name
            link_names = [l.name for l in self.env.left_arm.get_links()]
            # Heuristic to find the end effector
            if "Link_5" in link_names:
                self.ee_link_name = "Link_5"
            elif "link6" in link_names:
                self.ee_link_name = "link6"
            else:
                # Fallback to the last link
                self.ee_link_name = link_names[-1]
                
            self.kinematics = SimpleKinematics(self.env.left_arm, self.ee_link_name)

    def get_action(self, obs):
        self._init_kinematics()
        
        # 1. Get Privileged Info (Object Position)
        # Find the red block
        target_actor = None
        for actor in self.env.scene.get_all_actors():
            # In setup_scene_lift, we add a block. It usually gets a default name or we can find it by visual
            # But here we just look for the last added actor which is likely the block
            # Or we can check the user data if we set it.
            # Since we didn't set names explicitly in add_block (it returns actor), 
            # we can assume the block is one of the actors with "box" geometry or just the last one.
            # Let's try to find the actor that is NOT the ground or robot.
            if actor.name and "actor" in actor.name: # SAPIEN default names are actor_X
                 # Simple heuristic: check height. Block is at z ~ 0.015
                 pos = actor.get_pose().p
                 if 0.01 < pos[2] < 0.05:
                     target_actor = actor
                     break
        
        if target_actor is None:
            # Fallback if we can't find it, just stay put
            return np.zeros(14)

        block_pos = target_actor.get_pose().p
        
        current_qpos = self.env.left_arm.get_qpos()
        target_qpos = current_qpos.copy()
        
        # Gripper orientation: Pointing down.
        # SO-100/101 arm: 
        # We need the end-effector Z axis to point to World -Z.
        # And we want the gripper jaws to align with the block if possible, but for a cube it matters less.
        # Let's define a fixed downward rotation.
        # This depends on the zero-pose of the arm.
        # Usually, a rotation of 90 deg around Y or similar is needed.
        # Let's try a standard downward quaternion.
        # We can use scipy to generate it.
        # Looking down: x-axis forward, z-axis down? Or z-axis forward (gripper direction)?
        # Usually for robotic arms, Z is the approach vector.
        # So we want Z to be [0, 0, -1].
        
        # Let's try:
        rot = R.from_euler('xyz', [0, np.pi/2, 0]) # Rotate 90 deg around Y to point X-axis down? 
        # No, usually Z is the axis of the last link.
        # If Z is the axis of rotation of the last joint (wrist roll), then X is usually the approach.
        # Let's assume standard convention: X is approach.
        # Then we want X = [0, 0, -1].
        # Let's try a known good orientation for picking from top.
        target_quat = [0, 0.707, 0, 0.707] # Example quaternion
        
        # Better: use look_at logic or hardcoded euler
        # Euler: [0, pi/2, 0] often works for 5-DOF arms to look down
        target_quat = R.from_euler('xyz', [0, np.pi/2, 0]).as_quat()
        target_quat = [target_quat[3], target_quat[0], target_quat[1], target_quat[2]] # wxyz

        if self.stage == "APPROACH":
            # Target: 10cm above block
            target_pos = block_pos + np.array([0, 0, 0.12])
            target_pose = sapien.Pose(target_pos, target_quat)
            target_qpos = self.kinematics.compute_ik(target_pose)
            self.gripper_state = 1.0 # Open
            
            # Check error
            ee_pose = self.env.left_arm.get_links()[self.kinematics.ee_link_idx].get_pose()
            dist = np.linalg.norm(ee_pose.p - target_pos)
            if dist < 0.02:
                self.stage = "DOWN"
            
        elif self.stage == "DOWN":
            target_pos = block_pos + np.array([0, 0, 0.02]) # Grasp height
            target_pose = sapien.Pose(target_pos, target_quat)
            target_qpos = self.kinematics.compute_ik(target_pose)
            self.gripper_state = 1.0
            
            ee_pose = self.env.left_arm.get_links()[self.kinematics.ee_link_idx].get_pose()
            dist = np.linalg.norm(ee_pose.p - target_pos)
            if dist < 0.02:
                self.stage = "GRASP"
            
        elif self.stage == "GRASP":
            target_qpos = current_qpos # Stay
            self.gripper_state = -1.0 # Close
            # Wait a bit? In script we just return action.
            # We can check if gripper is closed (joint state)
            # For now, just transition after a few steps?
            # Let's assume we stay in GRASP for 1 step then LIFT (sim will handle physics)
            # Better: check if gripper joints are moving?
            self.stage = "LIFT"
            
        elif self.stage == "LIFT":
            target_pos = block_pos + np.array([0, 0, 0.2]) # Lift up
            target_pose = sapien.Pose(target_pos, target_quat)
            target_qpos = self.kinematics.compute_ik(target_pose)
            self.gripper_state = -1.0
            
        # Combine arm qpos + gripper
        # Map gripper_state to joint value. 
        # Open = 0.04 (m), Closed = 0.0
        # If gripper_state is 1.0 -> 0.04, -1.0 -> 0.0
        gripper_val = 0.04 if self.gripper_state > 0 else 0.0
        
        # The arm has 5 joints + 1 gripper (usually 2 fingers mimic)
        # We need to know the DOF of the arm.
        # SO-100 is usually 5 DOF + gripper.
        # Let's assume the last element of qpos is gripper.
        
        full_action = np.zeros(14) # 7 for left, 7 for right
        
        # Left arm action
        # We computed IK for the arm joints.
        # If IK returns 5 values, we append gripper.
        # If IK returns 6 values (including gripper), we overwrite gripper.
        
        # Assuming IK returns just arm joints (5 or 6)
        # We need to construct the full 7-dim vector for the left arm controller
        # (assuming the gym env expects 7 dims per arm)
        
        # Let's pad or truncate target_qpos to match 6 dims (5 arm + 1 gripper)
        # If robot has 5 arm joints:
        if len(target_qpos) == 5:
             left_action = np.concatenate([target_qpos, [gripper_val]])
        else:
             # If IK included gripper or something else, we overwrite the last one
             left_action = target_qpos
             left_action[-1] = gripper_val
             
        # Pad to 7 if needed (Gym env defined 7 dims)
        if len(left_action) < 7:
            left_action = np.concatenate([left_action, np.zeros(7 - len(left_action))])
            
        full_action[:7] = left_action
        
        return full_action
