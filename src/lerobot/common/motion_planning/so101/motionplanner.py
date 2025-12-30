import mplib
import numpy as np
import sapien
from transforms3d import euler

from lerobot.common.motion_planning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver

class SO101ArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    OPEN = 0.6 # Joint value for open
    CLOSED = -0.5 # Joint value for closed (approx)
    MOVE_GROUP = "moving_jaw_so101_v1_link"

    def __init__(
        self,
        env,
        robot: sapien.physx.PhysxArticulation,
        urdf_path: str,
        tcp_link_name: str,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,
        visualize_target_grasp_pose: bool = True,
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
        self._so_101_visual_grasp_pose_transform = sapien.Pose(q=euler.euler2quat(0, 0, np.pi / 2))
        self.grasp_pose_visual = None # TODO: Add visualization if needed

    @property
    def _so_101_grasp_pose_tcp_transform(self):
        # self.robot.find_link_by_name("gripper_link_tip").pose * tcp_pose.inv()
        # But we don't have env.agent.tcp_pose.
        # We assume tcp_link_name IS the TCP.
        # If "gripper_link_tip" is the TCP, then this transform is Identity.
        
        # In the original code:
        # self.base_env.agent.robot.links_map["gripper_link_tip"].pose.sp * self.base_env.agent.tcp_pose.sp.inv()
        
        # If we pass "gripper_link_tip" as tcp_link_name, then this is Identity.
        # Let's assume tcp_link_name is "gripper_link_tip".
        return sapien.Pose()

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target * self._so_101_visual_grasp_pose_transform)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        # If TCP is gripper_link_tip, we might not need offset.
        # But original code had it.
        return sapien.Pose(p=target.p + self._so_101_grasp_pose_tcp_transform.p, q=target.q)

    def close_gripper(self, t=15, gap=0.1):
        """
        缓慢闭合夹爪，gap=0完全闭合，gap=1完全张开，gap=0.5张开一半。
        """
        qpos = self.robot.get_qpos()
        start = qpos[-1]
        # gap=0: end=CLOSED, gap=1: end=OPEN
        end = self.CLOSED * (1 - gap) + self.OPEN * gap
        for i in range(t):
            alpha = (i + 1) / t
            qpos_step = qpos.copy()
            qpos_step[-1] = (1 - alpha) * start + alpha * end
            left_qpos = self.env.left_arm.get_qpos()
            right_qpos = self.env.right_arm.get_qpos()
            if self.robot == self.env.left_arm:
                left_qpos = qpos_step
            elif self.robot == self.env.right_arm:
                right_qpos = qpos_step
            action = np.concatenate([left_qpos, right_qpos])
            self.env.step(action)
            if self.vis:
                self.env.scene.step()

    def open_gripper(self, t=15):
        qpos = self.robot.get_qpos()
        qpos[-1] = self.OPEN
        
        for _ in range(t):
            left_qpos = self.env.left_arm.get_qpos()
            right_qpos = self.env.right_arm.get_qpos()
            
            if self.robot == self.env.left_arm:
                left_qpos = qpos
            elif self.robot == self.env.right_arm:
                right_qpos = qpos
            
            action = np.concatenate([left_qpos, right_qpos])
            self.env.step(action)
