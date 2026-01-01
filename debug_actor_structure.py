#!/usr/bin/env python3
"""Debug script to understand ManiSkill Actor structure."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "grasp-cube-sample")

import numpy as np
import sapien
import sapien.physx as physx
from lerobot.envs.sapien_env import SO101TaskEnv

# Create environment
env = SO101TaskEnv(num_envs=1, obs_mode="rgb", control_freq=10, render_mode="human")
env.reset(seed=0)

# Get the cube actor
cube = env.task_actors[0]

print(f"\n=== Cube Actor Structure ===")
print(f"Type: {type(cube)}")
print(f"Cube: {cube}")
print(f"Dir (attributes): {[x for x in dir(cube) if not x.startswith('_')]}")

# Check for entity/entities
print(f"\nhas 'entity': {hasattr(cube, 'entity')}")
print(f"has 'entities': {hasattr(cube, 'entities')}")
print(f"has '_objs': {hasattr(cube, '_objs')}")

if hasattr(cube, 'entity'):
    print(f"\ncube.entity type: {type(cube.entity)}")
    print(f"cube.entity: {cube.entity}")
    entity = cube.entity
elif hasattr(cube, 'entities'):
    print(f"\ncube.entities: {cube.entities}")
    entity = cube.entities[0] if cube.entities else None
else:
    entity = None

if entity is not None:
    print(f"\n=== Entity Components ===")
    print(f"Entity type: {type(entity)}")
    print(f"has 'find_component_by_type': {hasattr(entity, 'find_component_by_type')}")
    print(f"has 'get_components': {hasattr(entity, 'get_components')}")
    
    if hasattr(entity, 'get_components'):
        comps = list(entity.get_components())
        print(f"Total components: {len(comps)}")
        for i, comp in enumerate(comps):
            print(f"  [{i}] {type(comp).__name__}: {comp}")
            if isinstance(comp, (physx.PhysxRigidDynamicComponent, physx.PhysxRigidStaticComponent)):
                print(f"      ^ This is a rigid component!")
    
    if hasattr(entity, 'find_component_by_type'):
        dyn = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
        stat = entity.find_component_by_type(physx.PhysxRigidStaticComponent)
        print(f"find_component_by_type(PhysxRigidDynamicComponent): {dyn}")
        print(f"find_component_by_type(PhysxRigidStaticComponent): {stat}")

if hasattr(cube, '_objs'):
    print(f"\n=== cube._objs ===")
    print(f"Type: {type(cube._objs)}")
    print(f"Length: {len(cube._objs)}")
    for i, obj in enumerate(cube._objs):
        print(f"  [{i}] {type(obj).__name__}: {obj}")
        if hasattr(obj, 'get_components'):
            comps = list(obj.get_components())
            print(f"      Components: {len(comps)}")
            for j, comp in enumerate(comps):
                print(f"        [{j}] {type(comp).__name__}")

env.close()
