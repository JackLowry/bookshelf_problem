from isaacgym import gymapi
from math import pi, sqrt


gym = gymapi.acquire_gym()
# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

shelf_asset_options = gymapi.AssetOptions()
panda_asset = gym.load_asset(sim, "/home/aurmr/isaacgym/aurmr_grasping_research/assets/", "Panda/panda.urdf", asset_options)
shelf_asset = gym.load_asset(sim, "/home/aurmr/isaacgym/aurmr_grasping_research/assets/", "shelf/shelf.urdf", shelf_asset_options)
box_asset = gym.load_asset(sim, "/home/aurmr/isaacgym/aurmr_grasping_research/assets/", "shelf/box.urdf", gymapi.AssetOptions())
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

env = gym.create_env(sim, lower, upper, 8)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

panda_handle = gym.create_actor(env, panda_asset, pose, "Panda", 0, 0)

pose.p = gymapi.Vec3(-1, 0, 1)
pose.r = gymapi.Quat(0, 1/sqrt(2), 0.0, 1/sqrt(2))
shelf_handle = gym.create_actor(env, shelf_asset, pose, "Shelf", 0, 0)
# print(gym.set_actor_scale(env, shelf_handle, 1))

pose.p = gymapi.Vec3(-1.0, 0, .2)
pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
box_asset = gym.create_actor(env, box_asset, pose, "Box", 0, 0)

props = gym.get_actor_dof_properties(env, panda_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(1000.0)
props["damping"].fill(200.0)
gym.set_actor_dof_properties(env, panda_handle, props)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

while not gym.query_viewer_has_closed(viewer):

    # print(gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_ALL))

    gym.set_actor_dof_position_targets(env, panda_handle, [pi, 0 ,0, 0, 0, 0, pi, 0, pi/2])

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)