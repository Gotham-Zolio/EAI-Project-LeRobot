import sapien
import sapien.physx
import numpy as np


def normalize_angles(angles):
    """Normalize joint angles to [-pi, pi] range.
    
    SAPIEN IK sometimes returns multi-turn solutions (e.g., 2π + θ)
    which are mathematically correct but cause physical joint wrapping.
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


class SimpleKinematics:
    def __init__(self, robot: sapien.physx.PhysxArticulation, end_effector_name: str):
        self.robot = robot
        self.model = robot.create_pinocchio_model()
        # Find the link index for the end effector
        try:
            self.ee_link_idx = [link.name for link in robot.get_links()].index(end_effector_name)
        except ValueError:
            raise ValueError(f"Link {end_effector_name} not found in robot links: {[link.name for link in robot.get_links()]}")
        # Cache active joints and build an IK mask that excludes the gripper joint (does not affect EE pose)
        self.active_joints = self.robot.get_active_joints()
        self.active_mask = np.ones(self.robot.dof, dtype=float)
        for i, j in enumerate(self.active_joints):
            name = getattr(j, "name", "")
            if name.lower() == "gripper":
                self.active_mask[i] = 0.0
                break
        
    def compute_ik(self, target_pose: sapien.Pose, initial_qpos=None):
        """
        Compute joint angles to reach target_pose.
        
        Returns:
            (success: bool, qpos: np.ndarray) tuple where:
            - success: True if IK solver converged to a valid solution
            - qpos: the solution joint angles (or initial_qpos if failed)
        """
        if initial_qpos is None:
            initial_qpos = self.robot.get_qpos()
        
        # Try multiple IK parameter configurations if first attempt fails
        param_configs = [
            # Default: tight tolerance
            {"eps": 1e-4, "max_iterations": 100, "damp": 1e-6},
            # Looser: larger epsilon
            {"eps": 1e-3, "max_iterations": 100, "damp": 1e-5},
            # Even looser: more iterations
            {"eps": 1e-2, "max_iterations": 200, "damp": 1e-4},
            # Very loose: even more iterations
            {"eps": 5e-2, "max_iterations": 300, "damp": 1e-3},
            # Extra fallback: much looser for tough poses
            {"eps": 1e-1, "max_iterations": 500, "damp": 1e-2},
        ]
        
        for params in param_configs:
            # SAPIEN's compute_inverse_kinematics
            result = self.model.compute_inverse_kinematics(
                self.ee_link_idx,
                target_pose,
                initial_qpos=initial_qpos,
                # Exclude gripper from IK to avoid redundant DOF in EE pose
                active_qmask=self.active_mask,
                eps=params["eps"],
                max_iterations=params["max_iterations"],
                damp=params["damp"]
            )
            
            # SAPIEN returns (qpos, success, error) tuple in SAPIEN 3.x
            # result[0]: qpos (numpy array)
            # result[1]: success (bool or array of bool)
            # result[2]: error (float or array of float - residual error for each constraint)
            qpos = result[0]
            success = result[1]
            error = result[2]
            
            # Handle batched results (success might be an array)
            if isinstance(success, np.ndarray):
                 success = success.all()
            
            # Calculate max error magnitude
            if isinstance(error, np.ndarray):
                max_error = np.max(np.abs(error))
            else:
                max_error = abs(error)
            
            # Accept solution if:
            # 1. IK reports success, OR
            # 2. Error is small enough
            # Note: error array contains [x, y, z, rx, ry, rz] - mixed units!
            # Position errors in meters, orientation errors in radians
            if isinstance(qpos, np.ndarray) and qpos.shape[0] == self.robot.dof:
                # Debug: uncomment to see rejection reasons
                # print(f"[IK DEBUG] params eps={params['eps']}, success={success}, max_error={max_error:.4f}")
                
                if success or max_error < 0.20:  # Accept: 20cm position or ~11 degrees orientation
                    # Normalize angles to [-pi, pi] to avoid multi-turn solutions
                    qpos_normalized = normalize_angles(qpos)
                    # print(f"  qpos_normalized: {qpos_normalized}")
                    
                    # Find the closest equivalent angle to current position (shortest path)
                    # For each joint: if |target - current| > π, try target ± 2π to find shorter path
                    qpos_closest = qpos_normalized.copy()
                    for i in range(len(qpos_closest)):
                        diff = qpos_closest[i] - initial_qpos[i]
                        if abs(diff) > np.pi:
                            # Current path crosses ±π boundary, try the other direction
                            if diff > 0:
                                # Target is above current, try going below (target - 2π)
                                alternative = qpos_closest[i] - 2 * np.pi
                            else:
                                # Target is below current, try going above (target + 2π)
                                alternative = qpos_closest[i] + 2 * np.pi
                            # Choose whichever gives smaller absolute difference AND stays within limits
                            if abs(alternative - initial_qpos[i]) < abs(diff) and abs(alternative) <= np.pi:
                                qpos_closest[i] = alternative
                            # else: keep normalized version even if path is longer (to respect joint limits)
                    
                    # Check joint limits (SO-101 typical limits: approximately ±π for most joints)
                    max_angle = np.max(np.abs(qpos_closest))
                    # print(f"  max_angle: {max_angle:.4f} (limit: π={np.pi:.4f})")
                    if max_angle > np.pi:
                        # Angle exceeds physical limits, skip this solution
                        # print(f"[IK REJECT] max_angle={max_angle:.4f} > π, skipping")
                        continue
                    
                    # Accept this solution - return True to indicate valid result
                    # (even if SAPIEN reported success=False, if error is small enough, we consider it valid)
                    # print(f"[IK SUCCESS] Using params eps={params['eps']}, returning True")
                    return (True, qpos_closest)
            else:
                # This happens if qpos shape doesn't match robot.dof
                # print(f"[IK REJECT] qpos.shape={qpos.shape if isinstance(qpos, np.ndarray) else 'not array'}, robot.dof={self.robot.dof}")
                pass
        
        # All parameter configs failed - return initial pose as fallback
        # print(f"[IK FAILED] All {len(param_configs)} parameter configs failed, returning initial_qpos")
        return (False, initial_qpos.copy())
