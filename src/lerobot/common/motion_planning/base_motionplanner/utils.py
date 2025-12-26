import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from transforms3d import quaternions
# from mani_skill.utils.structs import Actor
# from mani_skill.utils import common
# from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

def get_component_mesh(component, to_world_frame=True):
    # Simplified version for SAPIEN 3
    # Assuming component is PhysxRigidDynamicComponent or similar
    # We need to extract collision meshes or visual meshes.
    # Usually collision meshes are boxes/convex meshes.
    
    # For the cube in LeRobotGymEnv, it's a box.
    # Let's try to get the collision shapes.
    if isinstance(component, sapien.Entity):
        # If passed an entity/actor, find the component
        component = component.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if component is None:
             component = component.find_component_by_type(physx.PhysxRigidStaticComponent)
    
    if component is None:
        return None

    # This is a bit hacky without full trimesh_utils. 
    # For a simple box actor created with add_box, we can reconstruct it.
    # But `get_actor_obb` expects a mesh.
    
    # Let's assume for now we are dealing with the blocks created in sapien_env.py
    # They are boxes.
    
    # If we can't easily get the mesh, we can construct a box mesh if we know the size.
    # But `get_actor_obb` is used to find the OBB.
    
    # Alternative: Use SAPIEN's AABB and create a box from it.
    # entity.get_global_aabb_fast() returns min/max.
    
    # But `compute_grasp_info_by_obb` needs an Oriented Bounding Box (trimesh).
    
    # Let's try to use the collision shapes.
    shapes = component.get_collision_shapes()
    if len(shapes) > 0:
        shape = shapes[0]
        if isinstance(shape, physx.PhysxCollisionShapeBox):
            half_size = shape.half_size
            pose = shape.local_pose
            # Transform to world
            # component.entity.pose * pose
            
            # Create trimesh box
            box = trimesh.creation.box(extents=half_size*2)
            
            # Apply transform
            world_pose = component.entity.pose * pose
            box.apply_transform(world_pose.to_transformation_matrix())
            return box
            
    return None

def get_actor_obb(actor, to_world_frame=True, vis=False):
    # actor is sapien.Entity
    mesh = get_component_mesh(actor, to_world_frame=to_world_frame)
    if mesh is None:
        # Fallback: AABB
        # print(f"Warning: Could not get mesh for {actor.name}, using AABB")
        aabb_min, aabb_max = actor.find_component_by_type(physx.PhysxRigidDynamicComponent).get_global_aabb_fast()
        extents = aabb_max - aabb_min
        center = (aabb_max + aabb_min) / 2
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(center)
        return box.bounding_box_oriented

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box."""
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)
    approaching = approaching / np.linalg.norm(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = closing / np.linalg.norm(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info
