#!/home/aurmr/anaconda3/envs/rlgpu3/bin/python

"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from turtle import end_poly
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math

from pyparsing import line
import numpy as np
import torch
import random
import time

from util import backproject_tensor, get_rect_planes, match_planes

BOX_SEG_ID = 1
NUM_ENVS = 8
def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def batch_tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    end_w = torch.linspace(0, 1, steps=steps).to(start)

    start = start.unsqueeze(-1).expand(-1, -1, steps)
    end = end.unsqueeze(-1).expand(-1, -1, steps)

    out = start_w * start + end_w * end

    return out

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default":NUM_ENVS, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer_prop = gymapi.CameraProperties()
viewer_prop.use_collision_geometry = True
viewer = gym.create_viewer(sim, viewer_prop)
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "/home/aurmr/research/isaac_manip/assets"

gym.set_light_parameters(sim, 3, gymapi.Vec3(.3, .3, .3), gymapi.Vec3(.3, .3, .3), gymapi.Vec3(1, 0, 1))

# create shelf asset
SHELF_LENGTH = 1.25
SHELF_WALL_WIDTH = .025
SHELF_DEPTH = 0.3

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
shelf_asset = gym.load_asset(sim, asset_root, "shelf/shelf.urdf", asset_options)

shelf_bottom_idx = gym.find_asset_rigid_body_index(shelf_asset, "bottom")
shelf_sensor_pose = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
shelf_sensor_idx = gym.create_asset_force_sensor(shelf_asset, shelf_bottom_idx, shelf_sensor_pose)


# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda_no_visual.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)
    franka_dof_props["velocity"][:7].fill(0.05)
    # default 87 87 87 87 12 12 12
    franka_dof_props["effort"][:7].fill(150)
else:       # osc
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["ball_EEF"]

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)
franka_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi)


shelf_pose = gymapi.Transform() 
shelf_pose.p = gymapi.Vec3(-.6, 0, 1.25)
shelf_pose.r = gymapi.Quat(0, 0, 0, 1)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)



envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []
seg_tensors = []
depth_tensors = []
proj_matrixes = []
inv_view_matrixes = []
camera_idxs = []
box_size_tensor = []
box_sizes = []
box_assets = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add shelf
    shelf_handle = gym.create_actor(env, shelf_asset, shelf_pose, "Shelf", i, 0)
    # num_sensors = gym.get_actor_force_sensor_count(env, shelf_handle)
    # for i in range(num_sensors):
    #     sensor = gym.get_actor_force_sensor(env, shelf_handle, i)
    
    # add box

    # create box asset

    
    BOX_SIZE_MINS = [.05, .05, .05]
    BOX_SIZE_MAXS = [.3, .3, .3]

    box_x = random.random()*(BOX_SIZE_MAXS[0]-BOX_SIZE_MINS[0])+BOX_SIZE_MINS[0]
    box_y = random.random()*(BOX_SIZE_MAXS[1]-BOX_SIZE_MINS[1])+BOX_SIZE_MINS[1]
    box_z = random.random()*(BOX_SIZE_MAXS[2]-BOX_SIZE_MINS[2])+BOX_SIZE_MINS[2]


    box_size = gymapi.Vec3(box_x, box_y, box_z)
    box_sizes.append(box_size)
    box_size_tensor.append(torch.Tensor([box_x, box_y, box_z]).to(device).unsqueeze(0))
    asset_options = gymapi.AssetOptions()
    asset_options.density = 110
    box_asset = gym.create_box(sim, box_size.x, box_size.y, box_size.z, asset_options)


    box_idx = gym.find_asset_rigid_body_index(box_asset, "box")
    box_sensor_pose = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
    box_sensor_idx = gym.create_asset_force_sensor(box_asset, box_idx, box_sensor_pose)

    BOX_OFFSET = 0.01
    bottom_handle = gym.find_actor_rigid_body_handle(env, shelf_handle, "bottom")
    bottom_pose = gym.get_rigid_transform(env, bottom_handle)
    box_pose = gymapi.Transform()
    box_pose.p.x = bottom_pose.p.x -SHELF_DEPTH/2+SHELF_WALL_WIDTH/2 + box_size.x/2 + BOX_OFFSET
    box_pose.p.y = bottom_pose.p.y -SHELF_LENGTH/2+SHELF_WALL_WIDTH/2 + box_size.y/2 + BOX_OFFSET
    box_pose.p.z = bottom_pose.p.z + SHELF_WALL_WIDTH/2 + box_size.z/2 + BOX_OFFSET
    box_pose.r = gymapi.Quat(0, 0, 0, 1)

    
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0, segmentationId=BOX_SEG_ID)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    sensor = gym.get_actor_force_sensor(env, box_handle, 0)


    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "ball_EEF")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "ball_EEF", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    #create camera sensor
    # add camera sensor
    camera_props = gymapi.CameraProperties()
    camera_props.width = 128
    camera_props.height = 128
    camera_props.enable_tensors = True
    camera_position = gymapi.Vec3(.3, 0, shelf_pose.p.z - SHELF_LENGTH/2+.5)
    lookat_position = gymapi.Vec3(shelf_pose.p.x, shelf_pose.p.y, shelf_pose.p.z - SHELF_LENGTH/2+.2)
    camera_handle = gym.create_camera_sensor(env, camera_props)
    camera_idxs.append(camera_handle)
    gym.set_camera_location(camera_handle, env, camera_position, lookat_position)

    # obtain camera tensor
    segmentation_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
    depth_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    # wrap camera tensor in a pytorch tensor
    torch_seg_tensor = gymtorch.wrap_tensor(segmentation_tensor).unsqueeze(0)
    seg_tensors.append(torch_seg_tensor)
    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor).unsqueeze(0)
    depth_tensors.append(torch_depth_tensor)
    # get camera matrix
    projection_matrix = torch.Tensor(gym.get_camera_proj_matrix(sim, env, camera_handle)).to(device)
    view_matrix = torch.Tensor(gym.get_camera_view_matrix(sim, env, camera_handle)).to(device)
    proj_matrixes.append(projection_matrix.unsqueeze(0))
    inv_view_matrix = torch.linalg.inv(view_matrix).to(device)
    inv_view_matrixes.append(inv_view_matrix.unsqueeze(0))

curr_proj_matrixes = torch.cat(proj_matrixes, 0)
curr_inv_view_matrixes = torch.cat(inv_view_matrixes, 0)
box_size_tensor = torch.cat(box_size_tensor, 0)
# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# # box corner coords, used to determine grasping yaw
# box_half_size = 0.5 * box_size
# corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
# corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 7, 1)
dof_vel = dof_states[:, 1].view(num_envs, 7, 1)

# Create a tensor noting whether the hand should return to the initial position
go_push = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

#TODO figure out how to include 3d

push_starts = []
push_ends = []
move_aways = []

for s in box_sizes:
    start = torch.Tensor((0, s.y*.25, s.z/2)).to(device)
    end = torch.Tensor((0, s.y*.25 + s.x*2.5, -s.z/2+SHELF_WALL_WIDTH)).to(device)
    away = torch.Tensor((0, s.y*2, -s.z/2+SHELF_WALL_WIDTH)).to(device)
    push_starts.append(start.unsqueeze(0))
    push_ends.append(end.unsqueeze(0))
    move_aways.append(away.unsqueeze(0))
push_starts = torch.cat(push_starts, 0)
push_ends = torch.cat(push_ends, 0)
move_aways = torch.cat(move_aways, 0)

last_dist = torch.zeros((num_envs)).to(device)

# step the physics
gym.simulate(sim)
gym.fetch_results(sim, True)

# refresh tensors
gym.refresh_rigid_body_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)
gym.refresh_jacobian_tensors(sim)
gym.refresh_mass_matrix_tensors(sim)

box_pos = rb_states[box_idxs, :3]

start_pos = box_pos.clone()
start_pos = start_pos + push_starts

end_pos = box_pos.clone()
end_pos = end_pos + push_ends

final_pos = box_pos.clone()
final_pos = final_pos + move_aways

env_origins = []
for i in range(num_envs):
    o = gym.get_env_origin(envs[::-1][i])
    o = torch.Tensor([o.x, o.y, o.z]).to(device)
    env_origins.append(o.unsqueeze(0))
env_origins = torch.cat(env_origins, 0)

cartesian_steps = 5
push_traj = batch_tensor_linspace(start_pos, end_pos, steps=cartesian_steps)
move_away_traj = batch_tensor_linspace(end_pos, final_pos + torch.rand(num_envs, 3).to(device)-.5, steps=cartesian_steps)
cartesian_idx = torch.zeros((num_envs), dtype=torch.long).to(device)
cartesian_steps = cartesian_steps*2
cartesian_traj = torch.cat((push_traj, move_away_traj), dim=2)
print(cartesian_idx)
# cartesian_traj = push_traj.view(1, 3, cartesian_steps).expand(num_envs, 3, cartesian_steps)
# simulation loop
face_colors = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]).unsqueeze(0).expand(num_envs, -1, -1)
while not gym.query_viewer_has_closed(viewer):
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    
    gym.clear_lines(viewer)
    # refresh camera images
    gym.render_all_camera_sensors(sim)

    #ALL CAMERA ACCESS CODE GOES IN HERE 
    gym.start_access_image_tensors(sim)

    # print(depth_tensors.dtype)
    # depth_seg = torch.where(seg_tensors == BOX_SEG_ID, depth_tensors.to(torch.double), 0.)

    curr_depth_tensors = torch.cat(depth_tensors, 0)
    curr_seg_tensors = torch.cat(seg_tensors, 0)

    seg_depth_tensors = torch.where(curr_seg_tensors==BOX_SEG_ID, curr_depth_tensors.to(torch.double), float('nan')).to(torch.float)
    # seg_depth_tensors = curr_depth_tensors

    # # print(curr_depth_tensors)
    # np.save("depth_image.npy", depth_image)

    pointcloud = backproject_tensor(curr_proj_matrixes, curr_inv_view_matrixes, seg_depth_tensors, curr_seg_tensors).clone()

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    
    box_face_normals, box_face_points = get_rect_planes(box_pos, box_rot, box_size_tensor)
    axes_box_face_points = box_face_points - env_origins.unsqueeze(1,).expand(-1, 6, -1)
    # face_idxs = torch.ones((pointcloud.shape[0], 6, pointcloud.shape[1])).to(torch.bool)
    gym.end_access_image_tensors(sim)
    #ALL CAMERA ACCESS CODE GOES IN HERE 
    num_lines = pointcloud.shape[1]
    camera_origin = torch.Tensor([camera_position.x, camera_position.y, camera_position.z]).to(device)
    
    for i in range(len(envs)):
        plane_origin = axes_box_face_points[i]
        plane_end = (axes_box_face_points[i] + box_face_normals[i])
        line_v = torch.zeros((plane_origin.shape[0]*2, 3)).to(device)
        env_origin = gym.get_env_origin(envs[::-1][i])
        env_origin = torch.Tensor([env_origin.x, env_origin.y, env_origin.z]).to(device)
        line_v[::2, :] = plane_origin
        line_v[1::2, :] = plane_end


        gym.add_lines(viewer, env, plane_origin.shape[0], line_v.cpu().numpy(), face_colors)

    box_face_points = box_face_points + env_origins.flip(0).unsqueeze(1,).expand(-1, 6, -1)
    face_idxs = match_planes(box_face_normals, box_face_points, pointcloud)

    # for i in range(pointcloud.shape[0]):
    #     pointcloud_e = pointcloud[i]

    #     non_nan_idxs = ~torch.isnan(pointcloud_e[:, 0])
    #     p_e = pointcloud_e[non_nan_idxs].unsqueeze(0).expand(6, -1, -1).clone()

    #     box_face_idxs = face_idxs[i][:, non_nan_idxs]
    #     face_normals = box_face_normals[i]
    #     colors_e = face_colors[i]
    #     env_origin = gym.get_env_origin(envs[i])
    #     p_env_origin = gym.get_env_origin(envs[::-1][i])

    #     box_face_idxs_normals_expand = face_normals.unsqueeze(1).expand(-1, box_face_idxs.shape[1], -1)
    #     line_normals = box_face_idxs_normals_expand[box_face_idxs, :]
    #     colors_expand = colors_e.unsqueeze(1).expand(-1, box_face_idxs.shape[1],-1)
    #     colors = colors_expand[box_face_idxs, :]
        
    #     final_p_e = p_e[box_face_idxs, :]

        
    #     env_origin = torch.Tensor([env_origin.x, env_origin.y, env_origin.z]).to(device)
    #     p_env_origin = torch.Tensor([p_env_origin.x, p_env_origin.y, p_env_origin.z]).to(device)
    #     line_start = torch.zeros((final_p_e.shape[0], 3)).to(device)
    #     line_end = torch.zeros((final_p_e.shape[0], 3)).to(device)
    #     line_origins = final_p_e - env_origin - p_env_origin
    #     line_start = line_origins
    #     line_end = line_origins+line_normals*.1
    #     lines = torch.cat((line_start, line_end), dim=1).clone()
    #     # line_v[1::2, :] = line_v[1::2, :]#+.1#line_v[1::2, :] + .1
    #     gym.add_lines(viewer, env, final_p_e.shape[0], lines.cpu().numpy(), colors)


    
    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # yaw_q = cube_grasping_yaw(box_rot, corners)
    # box_yaw_dir = quat_axis(yaw_q, 0)
    # hand_yaw_dir = quat_axis(hand_rot, 0)
    # yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box


    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box

    curr_goals = torch.index_select(cartesian_traj, 2, cartesian_idx)[:, :, 0]
    to_goal = curr_goals - hand_pos
    goal_dist = torch.norm(to_goal, dim=1)
    # hand_speed = torch.norm(hand_vel, dim=-1)
    delta_err = goal_dist - last_dist
    last_dist = goal_dist
    arrived = (goal_dist < 0.08) & (delta_err < 0.001)

    new_go_push = (~go_push & arrived).squeeze(-1)
    go_push = go_push | new_go_push

    cartesian_idx = torch.where(go_push & arrived & (cartesian_idx<(cartesian_steps-1)),  cartesian_idx+1, cartesian_idx)

    # compute goal position and orientation
    goal_pos = curr_goals
    goal_rot = init_rot

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # Deploy control based on type
    if controller == "ik":
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
    else:       # osc
        effort_action[:, :7] = control_osc(dpose)


    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))
    
    _fsdata = gym.acquire_force_sensor_tensor(sim)
    fsdata = gymtorch.wrap_tensor(_fsdata)
    # print(fsdata)   # force as Vec3
    # update viewer
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    gym.refresh_force_sensor_tensor(sim)
# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
