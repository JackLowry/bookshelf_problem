


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

from turtle import end_poly, left, right
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
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

from util import backproject_tensor, match_planes, batch_tensor_linspace, orientation_error, quaternion_rotation_matrix


class Bookshelf():
    
    BOX_SEG_ID = 1
    NUM_ENVS = 100
    DEVICE = "cuda:0"
    SIM_DEVICE_TYPE, COMPUTE_DEVICE_ID = gymutil.parse_device_str(DEVICE)
    GRAPHICS_DEVICE_ID = 0
    CONTROLLER = "ik"
    USE_GPU_PIPELINE = True
    USE_GPU = True
    PHYSICS_ENGINE = gymapi.SIM_PHYSX
    NUM_THREADS = 0

    ASSET_ROOT = "/home/aurmr/research/isaac_manip/assets"
    SHELF_LENGTH = 1.25
    SHELF_WALL_WIDTH = .025
    SHELF_DEPTH = 0.3

    DEBUG_VIZ = False

    
    def __init__(self):
        # set random seed
        np.random.seed(42)
        torch.set_printoptions(precision=4, sci_mode=False)

        # set torch device
        self.device = self.DEVICE

        self.init_gym()
        self.init_controller()
        self.create_assets()
        self.create_environments()

    def init_gym(self):
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.dt = 1.0/60.0
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = self.dt
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = self.USE_GPU_PIPELINE
        if self.PHYSICS_ENGINE == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.NUM_THREADS
            sim_params.physx.use_gpu = self.USE_GPU
        else:
            raise Exception("This example can only be used with PhysX")

        # create sim
        self.sim = self.gym.create_sim(self.COMPUTE_DEVICE_ID, self.GRAPHICS_DEVICE_ID, self.PHYSICS_ENGINE, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        viewer_prop = gymapi.CameraProperties()
        viewer_prop.use_collision_geometry = True
        self.viewer = self.gym.create_viewer(self.sim, viewer_prop)
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        self.gym.set_light_parameters(self.sim, 3, gymapi.Vec3(.3, .3, .3), gymapi.Vec3(.3, .3, .3), gymapi.Vec3(1, 0, 1))

    def init_controller(self):

        # Grab controller
        controller = self.CONTROLLER

        # Set controller parameters
        # IK params
        self.damping = 0.05

        # OSC params
        kp = 150.
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.
        kd_null = 2.0 * np.sqrt(kp_null)

    def create_assets(self):
        asset_root = self.ASSET_ROOT


        # create shelf asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.shelf_asset = self.gym.load_asset(self.sim, asset_root, "shelf/shelf.urdf", asset_options)

        shelf_bottom_idx = self.gym.find_asset_rigid_body_index(self.shelf_asset, "bottom")
        shelf_sensor_pose = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
        shelf_sensor_idx = self.gym.create_asset_force_sensor(self.shelf_asset, shelf_bottom_idx, shelf_sensor_pose)

        # load franka asset
        franka_asset_file = "urdf/franka_description/robots/franka_panda_no_visual.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        self.franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # configure franka dofs
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = self.franka_dof_props["lower"]
        franka_upper_limits = self.franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][:7].fill(400.0)
        self.franka_dof_props["damping"][:7].fill(40.0)
        self.franka_dof_props["velocity"][:7].fill(0.05)
        # default 87 87 87 87 12 12 12
        self.franka_dof_props["effort"][:7].fill(150)

        # grippers
        self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][7:].fill(800.0)
        self.franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = franka_mids[:7]

        self.default_franka_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        self.default_franka_dof_state["pos"] = self.default_dof_pos

        # send to torch
        default_dof_pos_tensor = to_torch(self.default_dof_pos, device=self.device)

    def get_rect_planes(self, box_trans, box_rot, box_sizes):
        self.box_orig_normals = torch.Tensor([[1, 0, 0], [-1, 0, 0],
                                [0, 1, 0], [0, -1, 0],
                                [0, 0, 1], [0, 0, -1]]).T.to(self.device).unsqueeze(0).expand(box_trans.shape[0], -1, -1)
        self.box_orig_points = torch.Tensor([[0.5, 0, 0, 1], [-0.5, 0, 0, 1],
                            [0, 0.5, 0, 1], [0, -0.5, 0, 1],
                            [0, 0, 0.5, 1], [0, 0, -0.5, 1]]).T.to(self.device).unsqueeze(0).expand(box_trans.shape[0], -1, -1)
        box_sizes = box_sizes.unsqueeze(-1)

        ones_row = torch.ones(box_sizes.shape[0], 1, 1).to(self.device)

        box_sizes_pad = torch.cat((box_sizes, ones_row), dim=1)


        self.box_orig_points = self.box_orig_points*box_sizes_pad


        rot_matrix = quaternion_rotation_matrix(box_rot) # envs x 3 x 3
        rot_90 = euler_angles_to_matrix(torch.Tensor([0, math.radians(90), 0]).to(self.device), "XYZ")
        rot_90 = rot_90.unsqueeze(0).expand(box_trans.shape[0], -1, -1)
        rot_matrix = torch.bmm(rot_90, rot_matrix)

        box_trans = box_trans.unsqueeze(2) # envs x 3 x 1
        trans_matrix = torch.cat((rot_matrix, box_trans), dim=2)
        homogenous_row = torch.Tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).expand(trans_matrix.shape[0], 1, -1).to(self.device)
        trans_matrix = torch.cat((trans_matrix, homogenous_row), dim=1)

        
        no_trans = torch.Tensor([[0, 0, 0]]).T.unsqueeze(0).expand(box_rot.shape[0], -1, -1).to(self.device)
        transform_90 = torch.cat((rot_90, no_trans), dim=2)
        transform_90 = torch.cat((transform_90, homogenous_row), dim=1)

        normals = torch.bmm(rot_matrix, self.box_orig_normals).transpose(1, 2)[:, :, 0:3] 
        points = torch.bmm(trans_matrix, self.box_orig_points).transpose(1, 2)[:, :, 0:3]
        return (normals, points)


    def calculate_graspability(self, box_normals, box_points, box_sizes):
        GRIPPER_APETURE = .2
        GRIPPER_FINGER_WIDTH = .05
        left_threshold = -(self.SHELF_LENGTH/2-self.SHELF_WALL_WIDTH/2)
        right_threshold = (self.SHELF_LENGTH/2-self.SHELF_WALL_WIDTH/2)
        camera_vector = -torch.Tensor([1, 0, 0]).to(self.device)
        left_vector = torch.Tensor([0, -1, 0]).to(self.device)
        right_vector = torch.Tensor([0, 1, 0]).to(self.device)

        # dot products to find the normals facing towards the camera, left
        facing_normal_idx = torch.argmin(torch.sum(box_normals*camera_vector,dim=2), 1)
        left_normal_idx = torch.argmin(torch.sum(box_normals*left_vector, dim=2), 1)
        facing_point = torch.index_select(box_points, 1, facing_normal_idx)[:, 0, :]
        facing_normal = torch.index_select(box_normals, 1, facing_normal_idx)[:, 0, :]
        facing_normal_2d = facing_normal[:, 0:2]
        camera_2d = camera_vector[0:2]
        dot_2d = torch.sum(facing_normal_2d*camera_2d, 1)
        norm_2d = torch.linalg.norm(facing_normal_2d, dim=1)*torch.linalg.norm(camera_2d)

        #clockwise rotation increase horizontal angle
        # range is 0-pi
        cos_horizontal_angle = -dot_2d/norm_2d
        horizontal_angle = torch.acos(cos_horizontal_angle)

        box_sizes = self.box_orig_points.transpose(1,2)[:, :, 0:3]
        box_size_horizontal = torch.index_select(box_sizes, 1, left_normal_idx)[0].abs().sum(1)
        too_wide = box_size_horizontal*2 > GRIPPER_APETURE

        edge_y_dist = box_size_horizontal*cos_horizontal_angle
        left_edge_y_pos = facing_point[:, 1] - edge_y_dist
        right_edge_y_pos = facing_point[:, 1] + edge_y_dist
        left_dist = left_edge_y_pos - left_threshold
        right_dist = right_threshold - right_edge_y_pos

        wall_dists = torch.where(left_dist < right_dist, left_dist, right_dist)
        space_to_grab = wall_dists > GRIPPER_FINGER_WIDTH
        #generate a gain from 0-1 which represents how much the box is facing the camera
        angle_dists = abs(horizontal_angle - math.pi/2)/math.pi/2
        # left_edge_dist = box_sizes[]

        wall_gain = 1
        angle_gain = 1
        # print(~too_wide)
        # print(space_to_grab)
        graspability = torch.where((~too_wide) & space_to_grab, wall_gain*wall_dists + angle_gain*angle_dists, torch.zeros(wall_dists.shape).to(self.device).to(torch.float))
        # print(graspability)
        return graspability

    def create_environments(self):
        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
        self.eef_index = franka_link_dict["ball_EEF"]

        # configure env grid
        num_envs = self.NUM_ENVS
        num_per_row = int(math.sqrt(num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % num_envs)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)
        franka_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi)


        shelf_pose = gymapi.Transform() 
        shelf_pose.p = gymapi.Vec3(-.6, 0, 1)
        shelf_pose.r = gymapi.Quat(0, 0, 0, 1)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        self.eef_idxs = []
        init_pos_list = []
        init_rot_list = []
        self.seg_tensors = []
        self.depth_tensors = []
        proj_matrixes = []
        inv_view_matrixes = []
        camera_idxs = []
        box_size_tensor = []
        self.box_sizes = []
        self.box_idxs = []
        self.franka_idxs = []
        self.box_actor_idxs = []
        for i in range(num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add shelf
            shelf_handle = self.gym.create_actor(env, self.shelf_asset, shelf_pose, "Shelf", i, 0)
            # num_sensors = self.gym.get_actor_force_sensor_count(env, shelf_handle)
            # for i in range(num_sensors):
            #     sensor = self.gymget_actor_force_sensor(env, shelf_handle, i)
            
            # # create box asset
            BOX_SIZE_MINS = [.05, .05, .05]
            BOX_SIZE_MAXS = [.3, .3, .2]
            # BOX_SIZE_MINS = [.1, .3, .2]
            # BOX_SIZE_MAXS = [.1, .3, .2]



            box_x = random.random()*(BOX_SIZE_MAXS[0]-BOX_SIZE_MINS[0])+BOX_SIZE_MINS[0]
            box_y = random.random()*(BOX_SIZE_MAXS[1]-BOX_SIZE_MINS[1])+BOX_SIZE_MINS[1]
            box_z = random.random()*(BOX_SIZE_MAXS[2]-BOX_SIZE_MINS[2])+BOX_SIZE_MINS[2]


            box_size = gymapi.Vec3(box_x, box_y, box_z)
            self.box_sizes.append(box_size)
            box_size_tensor.append(torch.Tensor([box_z, box_y, box_x]).to(self.device).unsqueeze(0))
            asset_options = gymapi.AssetOptions()
            asset_options.density = 50#110
            box_asset = self.gym.create_box(self.sim, box_size.x, box_size.y, box_size.z, asset_options)


            box_idx = self.gym.find_asset_rigid_body_index(box_asset, "box")
            box_sensor_pose = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
            box_sensor_idx = self.gym.create_asset_force_sensor(box_asset, box_idx, box_sensor_pose)

            # BOX_OFFSET = 0.01
            bottom_handle = self.gym.find_actor_rigid_body_handle(env, shelf_handle, "bottom")
            self.shelf_bottom_pose = self.gym.get_rigid_transform(env, bottom_handle)
            box_pose = gymapi.Transform()
            box_pose.p.x = 0#self.shelf_bottom_pose.p.x -self.SHELF_DEPTH/2+self.SHELF_WALL_WIDTH/2 + box_size.x/2 + BOX_OFFSET
            box_pose.p.y = 0#self.shelf_bottom_pose.p.y -self.SHELF_LENGTH/2+self.SHELF_WALL_WIDTH/2 + box_size.y/2 + BOX_OFFSET
            box_pose.p.z = 0#self.shelf_bottom_pose.p.z + self.SHELF_WALL_WIDTH/2 + box_size.z/2 + BOX_OFFSET
            box_pose.r = gymapi.Quat(0, 0, 0, 1)

            
            box_handle = self.gym.create_actor(env, box_asset, box_pose, "box", i, 0, segmentationId=self.BOX_SEG_ID)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            sensor = self.gym.get_actor_force_sensor(env, box_handle, 0)


            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)

            # add franka
            franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, self.franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, self.default_franka_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, self.default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "ball_EEF")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            eef_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "ball_EEF", gymapi.DOMAIN_SIM)
            self.eef_idxs.append(eef_idx)

            #create camera sensor
            # add camera sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = 128
            camera_props.height = 128
            camera_props.enable_tensors = True
            self.camera_position = gymapi.Vec3(.3, 0, shelf_pose.p.z - self.SHELF_LENGTH/2+.5)
            lookat_position = gymapi.Vec3(shelf_pose.p.x, shelf_pose.p.y, shelf_pose.p.z - self.SHELF_LENGTH/2+.2)
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            camera_idxs.append(camera_handle)
            self.gym.set_camera_location(camera_handle, env, self.camera_position, lookat_position)

            # obtain camera tensor
            segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
            depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)
            # wrap camera tensor in a pytorch tensor
            torch_seg_tensor = gymtorch.wrap_tensor(segmentation_tensor).unsqueeze(0)
            self.seg_tensors.append(torch_seg_tensor)
            torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor).unsqueeze(0)
            self.depth_tensors.append(torch_depth_tensor)
            # get camera matrix
            projection_matrix = torch.Tensor(self.gym.get_camera_proj_matrix(self.sim, env, camera_handle)).to(self.device)
            view_matrix = torch.Tensor(self.gym.get_camera_view_matrix(self.sim, env, camera_handle)).to(self.device)
            proj_matrixes.append(projection_matrix.unsqueeze(0))
            inv_view_matrix = torch.linalg.inv(view_matrix).to(self.device)
            inv_view_matrixes.append(inv_view_matrix.unsqueeze(0))

            box_idx = self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
            self.box_actor_idxs.append(torch.Tensor([box_idx]).to(torch.long).to(self.device))
            franka_idx = self.gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM)
            self.franka_idxs.append(torch.Tensor([franka_idx]).to(torch.long).to(self.device))


        # initial hand position and orientation tensors
        self.init_pos = torch.Tensor(init_pos_list).view(self.NUM_ENVS, 3).to(self.device)
        self.init_rot = torch.Tensor(init_rot_list).view(self.NUM_ENVS, 4).to(self.device)

        self.proj_matrixes = torch.cat(proj_matrixes, 0)
        self.inv_view_matrixes = torch.cat(inv_view_matrixes, 0)
        self.box_size_tensor = torch.cat(box_size_tensor, 0)            
        self.franka_idxs = torch.cat(self.franka_idxs, 0)
        self.box_actor_idxs = torch.cat(self.box_actor_idxs, 0)
        # point camera at middle env
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.face_colors = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]).unsqueeze(0).expand(self.NUM_ENVS, -1, -1)

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

        # downard axis
        self.down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.eef_index - 1, :, :7]

        # get mass matrix tensor
        # _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        # mm = gymtorch.wrap_tensor(_massmatrix)
        # mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = dof_states[:, 0].view(self.NUM_ENVS, 7, 1)
        dof_vel = dof_states[:, 1].view(self.NUM_ENVS, 7, 1)

        
        #reset franka
        self.default_franka_state = torch.zeros((self.default_dof_pos.shape[0], 2)).to(self.device)
        self.default_franka_state[:, 0] = torch.Tensor(self.default_dof_pos).unsqueeze(0)
        print(self.default_franka_state)
        self.default_franka_state = torch.tile(self.default_franka_state, (self.NUM_ENVS, 1))

    def control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.NUM_ENVS, 7)
        return u

    def reset(self):
        #TODO: fix this. Should take in a tensor with all the dof states, but works currently because only frankas have DOF states.
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.default_franka_state), gymtorch.unwrap_tensor(self.franka_idxs.to(torch.int)), self.NUM_ENVS)

        # Create a tensor noting whether the hand should return to the initial position
        self.go_push = torch.full([self.NUM_ENVS], False, dtype=torch.bool).to(self.device)

        env_origins = []
        for i in range(self.NUM_ENVS):
            o = self.gym.get_env_origin(self.envs[i])
            o = torch.Tensor([o.x, o.y, o.z]).to(self.device)
            env_origins.append(o.unsqueeze(0))
        self.env_origins = torch.cat(env_origins, 0)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        update_box_root_tensor = root_tensor.clone()
        push_starts = []
        push_ends = []
        move_aways = []


        POSSIBLE_ROTATIONS = [[0., 0., 0.], [1., 0., 0.],
                                           [0., 1., 0.], [1., 1., 0.],
                                           [0., 0., 1.], [1., 0., 1.]]

        for i in range(self.NUM_ENVS):
            env = self.envs[i]

            rotation = torch.Tensor(random.choice(POSSIBLE_ROTATIONS)).to(self.device)*(math.pi/2)

            rot_matrix = euler_angles_to_matrix(rotation, "XYZ")
            box_rot_matrix = euler_angles_to_matrix(rotation[[2, 1, 0]], "XYZ")
            rot_quat = matrix_to_quaternion(rot_matrix)
            rot_90 = euler_angles_to_matrix(torch.Tensor([0, math.radians(90), 0]).to(self.device), "XYZ")

            bottom_pose = self.shelf_bottom_pose
            s = self.box_sizes[i]
            s = torch.Tensor([s.x, s.y, s.z]).to(self.device)
            s = torch.abs(rot_matrix @ s)

            left_threshold = bottom_pose.p.y - self.SHELF_LENGTH/2+self.SHELF_WALL_WIDTH/2 + s[1]/2 
            right_threshold = bottom_pose.p.y + self.SHELF_LENGTH/2+self.SHELF_WALL_WIDTH/2 - s[1]/2
        
            state = torch.zeros((1, 13)).to(self.device)
            state[:, 0] = bottom_pose.p.x - self.SHELF_DEPTH/2+self.SHELF_WALL_WIDTH/2 + s[0]/2 
            state[:, 1] = left_threshold + (right_threshold-left_threshold)*random.random()
            state[:, 2] = bottom_pose.p.z + self.SHELF_WALL_WIDTH/2 + s[2]/2
            state[:, 3:7] = matrix_to_quaternion(box_rot_matrix)
            update_box_root_tensor[self.box_actor_idxs[i], :] = state

            start = torch.Tensor((0, s[1]*.35, s[2]/2)).to(self.device)
            end = torch.Tensor((0, s[1]*.35+ s[0]*2.5, -s[2]/2+self.SHELF_WALL_WIDTH)).to(self.device)
            away = torch.Tensor((0, s[1]*2, -s[2]/2+self.SHELF_WALL_WIDTH)).to(self.device)
            push_starts.append(start.unsqueeze(0))
            push_ends.append(end.unsqueeze(0))
            move_aways.append(away.unsqueeze(0))
        box_states =  gymtorch.unwrap_tensor(update_box_root_tensor)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, box_states, gymtorch.unwrap_tensor(self.box_actor_idxs.to(torch.int)), self.NUM_ENVS)
        push_starts = torch.cat(push_starts, 0)
        push_ends = torch.cat(push_ends, 0)
        move_aways = torch.cat(move_aways, 0)

        self.last_dist = torch.zeros((self.NUM_ENVS)).to(self.device)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        self.gym.clear_lines(self.viewer)
        # refresh camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

        box_pos = self.rb_states[self.box_idxs, :3]

        start_pos = box_pos.clone()
        start_pos = start_pos + push_starts

        end_pos = box_pos.clone()
        end_pos = end_pos + push_ends

        final_pos = box_pos.clone()
        final_pos = final_pos + move_aways

        cartesian_steps = 100
        push_traj = batch_tensor_linspace(start_pos, end_pos, steps=cartesian_steps)
        move_away_traj = batch_tensor_linspace(end_pos, final_pos + torch.rand(self.NUM_ENVS, 3).to(self.device)-.5, steps=cartesian_steps)
        self.cartesian_idx = torch.zeros((self.NUM_ENVS), dtype=torch.long).to(self.device)
        self.cartesian_steps = cartesian_steps*2
        self.cartesian_traj = torch.cat((push_traj, move_away_traj), dim=2)



    def observe(self):
         # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        self.gym.clear_lines(self.viewer)
        # refresh camera images
        self.gym.render_all_camera_sensors(self.sim)

        #ALL CAMERA ACCESS CODE GOES IN HERE 
        self.gym.start_access_image_tensors(self.sim)

        # print(depth_tensors.dtype)
        # depth_seg = torch.where(seg_tensors == BOX_SEG_ID, depth_tensors.to(torch.double), 0.)

        curr_depth_tensors = torch.cat(self.depth_tensors, 0)
        curr_seg_tensors = torch.cat(self.seg_tensors, 0)

        # seg_depth_tensors = torch.where(curr_seg_tensors==self.BOX_SEG_ID, curr_depth_tensors.to(torch.double), float('nan')).to(torch.float)

        pointcloud = backproject_tensor(self.proj_matrixes, self.inv_view_matrixes, curr_depth_tensors, curr_seg_tensors).clone()
        # the view matrix transforms to the global frame rather than the environment frame, so we have to subtract the environment origins to 
        # turn them into the environment frame
        pointcloud = pointcloud - self.env_origins.unsqueeze(1)

        segmentation_map = curr_seg_tensors.flatten(1, 2).unsqueeze(-1)
        state = torch.cat((pointcloud, segmentation_map), 2)

        if self.DEBUG_VIZ:
            self.box_pos = self.rb_states[self.box_idxs, :3]
            self.box_rot = self.rb_states[self.box_idxs, 3:7]
            box_face_normals, box_face_points = self.get_rect_planes(self.box_pos, self.box_rot, self.box_size_tensor)
            self.calculate_graspability(box_face_normals,box_face_points,self.box_size_tensor)
            axes_box_face_points = box_face_points
            self.gym.end_access_image_tensors(self.sim)
            #ALL CAMERA ACCESS CODE GOES IN HERE 
            for i in range(len(self.envs)):
                plane_origin = axes_box_face_points[i]
                plane_end = (axes_box_face_points[i] + box_face_normals[i])

                line_v = torch.zeros((plane_origin.shape[0]*2, 3)).to(self.device)

                line_v[::2, :] = plane_origin
                line_v[1::2, :] = plane_end
                self.gym.add_lines(self.viewer, self.envs[i], plane_origin.shape[0], line_v.cpu().numpy(), self.face_colors)

            box_face_points = box_face_points
            face_idxs = match_planes(box_face_normals, box_face_points, pointcloud)

            for i in range(pointcloud.shape[0]):
                pointcloud_e = pointcloud[i]

                non_nan_idxs = ~torch.isnan(pointcloud_e[:, 0])
                p_e = pointcloud_e[non_nan_idxs].unsqueeze(0).expand(6, -1, -1).clone()

                box_face_idxs = face_idxs[i][:, non_nan_idxs]
                face_normals = box_face_normals[i]
                colors_e = self.face_colors[i]

                box_face_idxs_normals_expand = face_normals.unsqueeze(1).expand(-1, box_face_idxs.shape[1], -1)
                line_normals = box_face_idxs_normals_expand[box_face_idxs, :]
                colors_expand = colors_e.unsqueeze(1).expand(-1, box_face_idxs.shape[1],-1)
                colors = colors_expand[box_face_idxs, :]
                
                final_p_e = p_e[box_face_idxs, :]

                line_start = torch.zeros((final_p_e.shape[0], 3)).to(self.device)
                line_end = torch.zeros((final_p_e.shape[0], 3)).to(self.device)
                line_origins = final_p_e #- env_origin - p_env_origin
                line_start = line_origins
                line_end = line_origins+line_normals*.1
                lines = torch.cat((line_start, line_end), dim=1).clone()
                self.gym.add_lines(self.viewer, self.envs[i], final_p_e.shape[0], lines.cpu().numpy(), colors)

        return state

    def step(self, action):
            # update viewer
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
            
            # self.gym.clear_lines(self.viewer)
            # # refresh camera images
            # self.gym.render_all_camera_sensors(self.sim)

    def do_actions(self, actions):
        #FIGURE OUT ACTION:
        time_max = int(3.0 / self.dt)
        steps = 0
        z_depth = .05
        while(steps < time_max):
            self.box_pos = self.rb_states[self.box_idxs, :3]
            self.box_rot = self.rb_states[self.box_idxs, 3:7]
            eef_pos = self.rb_states[self.eef_idxs, :3]
            eef_rot = self.rb_states[self.eef_idxs, 3:7]
            eef_vel = self.rb_states[self.eef_idxs, 7:]

            to_box = self.box_pos - eef_pos
            box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
            box_dir = to_box / box_dist
            box_dot = box_dir @ self.down_dir.view(3, 1)

            # determine if we have reached the initial position; if so allow the hand to start moving to the box

            # if hand is above box, descend to grasp offset
            # otherwise, seek a position above the box

            curr_goals = torch.index_select(self.cartesian_traj, 2, self.cartesian_idx)[:, :, 0]
            to_goal = curr_goals - eef_pos
            goal_dist = torch.norm(to_goal, dim=1)
            delta_err = goal_dist - self.last_dist
            self.last_dist = goal_dist
            arrived = (goal_dist < 0.06) & (delta_err < 0.001)

            new_go_push = (~self.go_push & arrived).squeeze(-1)
            self.go_push = self.go_push | new_go_push

            self.cartesian_idx = torch.where(self.go_push & arrived & (self.cartesian_idx<(self.cartesian_steps-1)),  self.cartesian_idx+1, self.cartesian_idx)

            # compute goal position and orientation
            goal_pos = curr_goals
            goal_rot = self.init_rot

            # compute position and orientation error
            pos_err = goal_pos - eef_pos
            orn_err = orientation_error(goal_rot, eef_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            # Deploy control based on type
            self.pos_action[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)

            # Deploy actions
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))
            
            _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
            fsdata = gymtorch.wrap_tensor(_fsdata)
            self.step()
            steps = steps+1
    def cleanup(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    bookshelf = Bookshelf()
    bookshelf.reset()
    frames = 0
    while not bookshelf.gym.query_viewer_has_closed(bookshelf.viewer):
        if frames % 200 == 0:
            bookshelf.reset()
            state = bookshelf.observe()
            print(state)
        bookshelf.step(None)
        frames = frames + 1
    bookshelf.cleanup()



# def quat_axis(q, axis=0):
#     basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
#     basis_vec[:, axis] = 1
#     return quat_rotate(q, basis_vec)



# def cube_grasping_yaw(q, corners):
#     """ returns horizontal rotation required to grasp cube """
#     rc = quat_rotate(q, corners)
#     yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
#     theta = 0.5 * yaw
#     w = theta.cos()
#     x = torch.zeros_like(w)
#     y = torch.zeros_like(w)
#     z = theta.sin()
#     yaw_quats = torch.stack([x, y, z, w], dim=-1)
#     return yaw_quats


# def tensor_linspace(start, end, steps=10):
#     """
#     Vectorized version of torch.linspace.
#     Inputs:
#     - start: Tensor of any shape
#     - end: Tensor of the same shape as start
#     - steps: Integer
#     Returns:
#     - out: Tensor of shape start.size() + (steps,), such that
#       out.select(-1, 0) == start, out.select(-1, -1) == end,
#       and the other elements of out linearly interpolate between
#       start and end.
#     """
#     assert start.size() == end.size()
#     view_size = start.size() + (1,)
#     w_size = (1,) * start.dim() + (steps,)
#     out_size = start.size() + (steps,)

#     start_w = torch.linspace(1, 0, steps=steps).to(start)
#     start_w = start_w.view(w_size).expand(out_size)
#     end_w = torch.linspace(0, 1, steps=steps).to(start)
#     end_w = end_w.view(w_size).expand(out_size)

#     start = start.contiguous().view(view_size).expand(out_size)
#     end = end.contiguous().view(view_size).expand(out_size)

#     out = start_w * start + end_w * end
#     return out




