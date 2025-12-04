import sapien
import numpy as np
from PIL import Image
import tyro

from sapien.pysapien.render import RenderCameraComponent

def add_camera(
    scene, name, width: int, height: int, fovx: float, fovy: float, near: float, far: float
) -> RenderCameraComponent:
    """
    Adds a camera to the given Sapien scene with specified parameters.
    Sapien does not have a direct method to add a camera using fovx and fovy,
    """
    camera_mount = sapien.Entity()
    camera = RenderCameraComponent(width, height)
    camera.set_fovx(fovx, compute_y=False)
    camera.set_fovy(fovy, compute_x=False)
    camera.near = near
    camera.far = far
    camera_mount.add_component(camera)
    scene.add_entity(camera_mount)
    camera_mount.name = name
    camera.name = name

    return camera


def render_scene(use_ray_tracing: bool = False, show_viewer: bool = True):
    """
    Sets up a basic Sapien scene with lights, geometry, a camera, 
    and renders an image, saving it to 'camera_view.png'.

    :param use_ray_tracing: If True, uses the ray tracing shader for high quality rendering.
    :param show_viewer: If True, opens an interactive viewer window.
    """
    
    # 1. Setup Rendering Environment
    if use_ray_tracing:
        sapien.render.set_viewer_shader_dir("rt")
        # Configure ray tracing settings
        sapien.render.set_ray_tracing_samples_per_pixel(64)
        sapien.render.set_ray_tracing_denoiser("optix")
    
    # 2. Scene Initialization
    scene = sapien.Scene()
    scene.set_timestep(1 / 240) # Set simulation timestep
    
    # Configure ground material and add ground plane
    ground_material = sapien.render.RenderMaterial()
    # Base color (light brown/tan)
    ground_material.base_color = np.array([202, 164, 114, 256]) / 256
    ground_material.specular = 0.5
    scene.add_ground(0, render_material=ground_material)

    # 3. Lighting Setup
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light(
        [0.5, 0, -1],  # Light direction
        color=[3.0, 3.0, 3.0], # Strong white light
        shadow=True,
        shadow_scale=2.0,
        shadow_map_size=4096,
    )
    
    # 4. Viewer Setup (for interactive viewing, although we take a picture)
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    
    # Initialize a 4x4 identity matrix for pose transformations
    mat44 = np.eye(4)
    
    # 5. Scene Geometry (Bounding Bands and Target Cube)
    
    # Builder for bounding bands (thin black planes)
    # Band half_size: [10, 0.009, 0.01] (1.8 cm total width)
    BAND_Z_POS = 0.005
    band_builder = scene.create_actor_builder()
    band_builder.add_box_visual(half_size=[10, 0.009, 0.01], material=[0, 0, 0])
    band_builder.add_box_collision(half_size=[10, 0.009, 0.01])
    
    # --- New Band Positions (Complex Sums from user query) ---
    
    # left_band: Y = 0.02 + 0.009 = 0.029
    left_band = band_builder.build(name="left_band")
    left_band.set_pose(sapien.Pose([0, 0.029, BAND_Z_POS]))
    
    # left_band_near: Y = 0.02 + 0.018 + 0.166 + 0.009 = 0.213
    left_band_near = band_builder.build(name="left_band_near")
    left_band_near.set_pose(sapien.Pose([0, 0.213, BAND_Z_POS]))

    # right_band_near: Y = 0.02 + 0.018 + 0.166 + 0.018 + 0.156 + 0.009 = 0.387
    right_band_near = band_builder.build(name="right_band_near")
    right_band_near.set_pose(sapien.Pose([0, 0.387, BAND_Z_POS]))
    
    # right_band: Y = 0.02 + 0.018 + 0.166 + 0.018 + 0.156 + 0.018 + 0.166 + 0.009 = 0.571
    right_band = band_builder.build(name="right_band")
    right_band.set_pose(sapien.Pose([0, 0.571, BAND_Z_POS]))
    
    # Builder for the Red Target Cube
    # Cube half_size: [0.015, 0.015, 0.015] (3 cm total size)
    CUBE_Z_POS = 0.015
    cube_builder = scene.create_actor_builder()
    cube_builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[1, 0, 0])
    cube_builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
    red_cube = cube_builder.build(name="red_cube")
    # Position the cube on the ground (z-coordinate must equal half_size_z)
    red_cube.set_pose(sapien.Pose([0.26, 0.3, CUBE_Z_POS]))
    
    # 6. Camera Setup and Placement
    
    # Camera intrinsic parameters (Field of View)
    camera = add_camera(
        scene=scene,
        name="camera",
        width=640,
        height=480,
        fovx=np.deg2rad(117.12),
        fovy=np.deg2rad(73.63),
        near=0.01,
        far=100,
    )
    
    # Define Camera Pose (Local Pose relative to its parent/world)
    
    cam_rot = np.array([
        [np.cos(np.pi/2), 0, np.sin(np.pi/2)],
        [0, 1, 0],
        [-np.sin(np.pi/2), 0, np.cos(np.pi/2)],
    ])
    mat44[:3, :3] = cam_rot
    
    # Translation: Height and new X, Y offsets
    cam_height = 0.407
    # New offset [26.0, 31.6] cm converted to meters: [0.26, 0.316, 0.407]
    cam_offset = np.array([0.26, 0.316, cam_height])
    mat44[:3, 3] = cam_offset

    # Apply the full 4x4 pose matrix to the camera
    camera.set_local_pose(sapien.Pose(mat44))
    
    # 7. Render and Save Image
    scene.step() # Advance simulation by one step
    scene.update_render() # Update render buffers
    camera.take_picture()    
    
    # Get color data and convert to 8-bit PIL image format
    rgba = camera.get_picture("Color")  # Output shape: [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    
    # Save the rendered image
    output_filename = "camera_view.png"
    rgba_pil.save(output_filename)
    print(f"\n✅ Scene rendered successfully and image saved to {output_filename}")
    
    if show_viewer:
        # Keep the viewer open for interaction
        while not viewer.closed:
            scene.step()
            scene.update_render()
            viewer.render()


if __name__ == "__main__":
    tyro.cli(render_scene)