import sapien
import numpy as np
import tyro
from sapien.pysapien.render import RenderCameraComponent

def create_scene(fix_root_link: bool = True, balance_passive_force: bool = True):
    # 1. Scene Initialization
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # 2. Ground and Table Setup
    ground_material = sapien.render.RenderMaterial()
    ground_material.base_color = np.array([202, 164, 114, 256]) / 256
    ground_material.specular = 0.5
    scene.add_ground(0, render_material=ground_material)

    # 3. Add Bounding Bands (from front_camera.py)
    BAND_Z_POS = 0.005
    band_builder = scene.create_actor_builder()
    band_builder.add_box_visual(half_size=[10, 0.009, 0.01], material=[0, 0, 0])
    band_builder.add_box_collision(half_size=[10, 0.009, 0.01])
    
    # Band positions
    band_positions = [0.029, 0.213, 0.387, 0.571]
    for i, y_pos in enumerate(band_positions):
        band = band_builder.build(name=f"band_{i}")
        band.set_pose(sapien.Pose([0, y_pos, BAND_Z_POS]))

    # 4. Load Robot (from robot.py)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    # Note: Ensure the assets folder is in the correct path relative to this script
    try:
        robot = loader.load("reference-scripts/assets/SO101/so101.urdf")
    except Exception:
        # Fallback if running from root and assets are in reference-scripts
        robot = loader.load("assets/SO101/so101.urdf")
        
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    
    # Set initial joint positions
    arm_init_qpos = [0, 0, 0, 0, 0]
    gripper_init_qpos = [0]
    robot.set_qpos(arm_init_qpos + gripper_init_qpos)

    # 5. Camera Setup (from front_camera.py)
    camera_mount = sapien.Entity()
    camera = RenderCameraComponent(640, 480)
    camera.set_fovx(np.deg2rad(117.12), compute_y=False)
    camera.set_fovy(np.deg2rad(73.63), compute_x=False)
    camera.near = 0.01
    camera.far = 100
    camera_mount.add_component(camera)
    camera_mount.name = "front_camera"
    scene.add_entity(camera_mount)

    # Camera Pose
    cam_rot = np.array([
        [np.cos(np.pi/2), 0, np.sin(np.pi/2)],
        [0, 1, 0],
        [-np.sin(np.pi/2), 0, np.cos(np.pi/2)],
    ])
    mat44 = np.eye(4)
    mat44[:3, :3] = cam_rot
    mat44[:3, 3] = np.array([0.26, 0.316, 0.407]) # Offset from front_camera.py
    camera_mount.set_pose(sapien.Pose(mat44))

    # 6. Viewer Setup
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-1, y=0.3, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    print("Simulation started. Close the viewer window to exit.")
    
    while not viewer.closed:
        for _ in range(4):
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
        viewer.render()

if __name__ == "__main__":
    tyro.cli(create_scene)
