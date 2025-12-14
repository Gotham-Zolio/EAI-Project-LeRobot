import sapien
import sapien.physx
import numpy as np

class SimpleKinematics:
    def __init__(self, robot: sapien.physx.PhysxArticulation, end_effector_name: str):
        self.robot = robot
        self.model = robot.create_pinocchio_model()
        # Find the link index for the end effector
        try:
            self.ee_link_idx = [link.name for link in robot.get_links()].index(end_effector_name)
        except ValueError:
            raise ValueError(f"Link {end_effector_name} not found in robot links: {[link.name for link in robot.get_links()]}")
        
    def compute_ik(self, target_pose: sapien.Pose, initial_qpos=None):
        """
        Compute joint angles to reach target_pose.
        """
        if initial_qpos is None:
            initial_qpos = self.robot.get_qpos()
            
        # SAPIEN's compute_inverse_kinematics
        result = self.model.compute_inverse_kinematics(
            self.ee_link_idx,
            target_pose,
            initial_qpos=initial_qpos,
            active_qmask=np.ones(self.robot.dof, dtype=float),
            eps=1e-4,
            max_iterations=100,
            damp=1e-6
        )
        
        if result[0]: # Success
            return result[1]
        else:
            # print("IK Failed")
            return initial_qpos # Return current pose if failed
