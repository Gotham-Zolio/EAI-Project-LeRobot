import mplib
import numpy as np
import sapien

from lerobot.common.motion_planning.base_motionplanner.motionplanner import BaseMotionPlanningSolver

class TwoFingerGripperMotionPlanningSolver(BaseMotionPlanningSolver):
    def __init__(
        self,
        env,
        robot: sapien.physx.PhysxArticulation,
        urdf_path: str,
        tcp_link_name: str,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(
            env,
            robot,
            urdf_path,
            tcp_link_name,
            debug,
            vis,
            base_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
        self.gripper_close = 0.0
        self.gripper_open = 0.04 # Default, override in subclass

    def close_gripper(self, t=15):
        # In LeRobotGymEnv, gripper is controlled by the last joint(s).
        # For SO101, it's likely the last joint.
        # We need to know which joints correspond to the gripper.
        # The `move_group` in planner is "gripper_link_tip" (link), but we need joints.
        
        # Assuming the last joint is the gripper joint for now, or we use named joints if we knew them.
        # SO101 has a specific gripper structure.
        
        # Let's look at SO101 subclass for specific values.
        pass

    def open_gripper(self, t=15):
        pass
