import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
from transforms3d import quaternions
from mani_skill.utils.structs import Actor
from mani_skill.utils import common
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh


def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    # ManiSkill Actor wraps entities in _objs list
    # Try to get the first object's rigid component
    rigid_comp = None
    
    # Try the standard path: actor._objs[0] with find_component_by_type
    try:
        if hasattr(actor, "_objs") and actor._objs:
            obj = actor._objs[0]
            if hasattr(obj, "find_component_by_type"):
                # Try dynamic first
                rigid_comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
                # Fallback to static
                if rigid_comp is None:
                    rigid_comp = obj.find_component_by_type(physx.PhysxRigidStaticComponent)
    except Exception as e:
        pass
    
    # If that didn't work, try entity path
    if rigid_comp is None:
        try:
            entity = actor.entity if hasattr(actor, "entity") else actor
            if hasattr(entity, "find_component_by_type"):
                rigid_comp = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
                if rigid_comp is None:
                    rigid_comp = entity.find_component_by_type(physx.PhysxRigidStaticComponent)
        except Exception:
            pass
    
    # Final fallback: iterate get_components if available
    if rigid_comp is None:
        try:
            entity = actor.entity if hasattr(actor, "entity") else actor
            if hasattr(entity, "get_components"):
                for comp in entity.get_components():
                    if isinstance(comp, (physx.PhysxRigidDynamicComponent, physx.PhysxRigidStaticComponent)):
                        rigid_comp = comp
                        break
        except Exception:
            pass
    
    assert rigid_comp is not None, f"Unable to find rigid component in actor: {actor}"
    mesh = get_component_mesh(rigid_comp, to_world_frame=to_world_frame)
    assert mesh is not None, f"Unable to get mesh for actor: {actor}"
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
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

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
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info