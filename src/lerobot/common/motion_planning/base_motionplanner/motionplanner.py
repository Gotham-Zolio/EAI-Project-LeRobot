import mplib
import numpy as np
import sapien
import trimesh

# from mani_skill.agents.base_agent import BaseAgent
# from mani_skill.envs.sapien_env import BaseEnv
# from mani_skill.utils.structs.pose import to_sapien_pose

def to_sapien_pose(pose):
    if isinstance(pose, sapien.Pose):
        return pose
    if isinstance(pose, (list, tuple, np.ndarray)):
        if len(pose) == 7:
            return sapien.Pose(p=pose[:3], q=pose[3:])
        elif len(pose) == 44: # 4x4 matrix flattened? No.
             pass
    return sapien.Pose(pose) # Fallback

class BaseMotionPlanningSolver:

    def __init__(
        self,
        env,
        robot: sapien.physx.PhysxArticulation,
        urdf_path: str,
        tcp_link_name: str,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.robot = robot
        self.urdf_path = urdf_path
        self.tcp_link_name = tcp_link_name
        
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose) if base_pose is not None else sapien.Pose()

        self.planner = self.setup_planner()
        # self.control_mode = self.base_env.control_mode 
        # LeRobotGymEnv uses joint position control by default in step()
        self.control_mode = "pd_joint_pos" 

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        # viewer = self.base_env.render_human()
        # LeRobotGymEnv doesn't expose render_human directly in the same way, 
        # but we can assume env.render() or just skip if not available.
        # For now, just pass or use input() if debug is strictly needed.
        if self.debug:
             input("Press Enter to continue...")

    def setup_planner(self):
        move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.urdf_path,
            srdf=self.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        return None

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            
            # LeRobotGymEnv step expects action for BOTH arms if it's the main env step.
            # But here we are controlling one robot.
            # We need to know if we are left or right arm, OR we need to construct the full action.
            # Since we passed `env` and `robot`, we can try to infer or just set the robot's drive target directly
            # and call scene.step(), bypassing env.step() if we want pure motion planning execution.
            # However, env.step() handles recording, rewards, etc.
            
            # For data collection, we probably want to use env.step().
            # But env.step() takes a combined action vector.
            
            # Strategy: Read current qpos of the OTHER arm, and combine.
            # This requires knowing which arm we are.
            
            # Let's assume we can set the robot's drive targets directly for now to ensure movement,
            # and then call env.step() with the current qpos of both arms as the "action".
            
            # Actually, better: Construct the full action vector.
            # self.env.left_arm and self.env.right_arm
            
            left_qpos = self.env.left_arm.get_qpos()
            right_qpos = self.env.right_arm.get_qpos()
            
            if self.robot == self.env.left_arm:
                left_qpos = qpos
            elif self.robot == self.env.right_arm:
                right_qpos = qpos
            
            action = np.concatenate([left_qpos, right_qpos])
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            # if self.vis:
            #     self.base_env.render_human()
        return obs, reward, terminated, truncated, info

        
    def move_to_pose_with_RRTStar(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos(), # .cpu().numpy()[0] if torch
            time_step=1/240, # self.base_env.control_timestep
            use_point_cloud=self.use_point_cloud,
            rrt_range=0.0,
            planning_time=1,
            planner_name="RRTstar",
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos(),
            time_step=1/240,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos(),
            time_step=1/240,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos(),
                time_step=1/240,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass
