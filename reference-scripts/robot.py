import sapien
import numpy as np
import tyro

def demo(fix_root_link: bool = True, balance_passive_force: bool = True):
    scene = sapien.Scene()
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot = loader.load("assets/SO101/so101.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    arm_init_qpos = [0, 0, 0, 0, 0]
    gripper_init_qpos = [0]
    init_qpos = arm_init_qpos + gripper_init_qpos
    robot.set_qpos(init_qpos)

    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.02, 0.02, 0.2], material=[1, 0, 0])
    builder.add_box_collision(half_size=[0.02, 0.02, 0.05])
    red_cube = builder.build(name="red_cube")
    red_cube.set_pose(sapien.Pose([0.32 + 0.105, 0, 0.1]))
    print("links names:", [link.name for link in robot.get_links()])
    # get link to attach camera
    camera_link = [link for link in robot.get_links() if "camera" in link.name][0]
    mounted_camera = scene.add_mounted_camera(
        name="mounted_camera",
        mount=camera_link.entity,
        pose=sapien.Pose(np.eye(4)),
        width=640,
        height=480,
        fovy=np.deg2rad(50),
        near=0.01,
        far=100,
    )

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
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
    tyro.cli(demo)