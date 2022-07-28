# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import random
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class Bookshelf(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 23
        num_acts = 9

        self.cfg["env"]["numObservations"] = 23
        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        BOX_SEG_ID = 1

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "/home/aurmr/reseach/isaac_manip/assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda_no_visual.urdf"
        shelf_asset_file = "shelf/shelf.urdf"


        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            shelf_asset_file = self.cfg["env"]["asset"].get("assetFileNameShelf", shelf_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # create shelf asset
        SHELF_LENGTH = 1.25
        SHELF_WALL_WIDTH = .025
        SHELF_DEPTH = 0.3

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        shelf_asset = self.gym.load_asset(self.sim, asset_root, "shelf/shelf.urdf", asset_options)

        shelf_bottom_idx = self.gym.find_asset_rigid_body_index(shelf_asset, "bottom")
        shelf_sensor_pose = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
        shelf_sensor_idx = self.gym.create_asset_force_sensor(shelf_asset, shelf_bottom_idx, shelf_sensor_pose)


        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(shelf_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(shelf_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num shelf bodies: ", self.num_cabinet_bodies)

        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:7].fill(400.0)
        franka_dof_props["damping"][:7].fill(40.0)
        franka_dof_props["velocity"][:7].fill(0.05)
        # default 87 87 87 87 12 12 12

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # send to torch
        default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)


        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        franka_hand_index = franka_link_dict["ball_EEF"]

        # configure env grid
        num_envs = self.num_envs
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

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_shelf_bodies = self.gym.get_asset_rigid_body_count(shelf_asset)
        num_shelf_shapes = self.gym.get_asset_rigid_shape_count(shelf_asset)
        num_box_bodies = 1  
        num_box_shapes = 1
        max_agg_bodies = num_franka_bodies + num_shelf_bodies + num_box_bodies
        max_agg_shapes = num_franka_shapes + num_shelf_bodies + num_box_bodies


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

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_pose, "franka", i, 2)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            self.gym.set_actor_dof_position_targets(env_ptr, franka_actor, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(self.env, franka_actor, "ball_EEF")
            hand_pose = self.gym.get_rigid_transform(self.env, hand_handle)
            init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(self.env, franka_actor, "ball_EEF", gymapi.DOMAIN_SIM)
            hand_idxs.append(hand_idx)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add shelf
            shelf_actor = self.gym.create_actor(self.env, shelf_asset, shelf_pose, "Shelf", i, 0)      

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            BOX_SIZE_MINS = [.05, .05, .05]
            BOX_SIZE_MAXS = [.3, .3, .2]

            box_x = random.random()*(BOX_SIZE_MAXS[0]-BOX_SIZE_MINS[0])+BOX_SIZE_MINS[0]
            box_y = random.random()*(BOX_SIZE_MAXS[1]-BOX_SIZE_MINS[1])+BOX_SIZE_MINS[1]
            box_z = random.random()*(BOX_SIZE_MAXS[2]-BOX_SIZE_MINS[2])+BOX_SIZE_MINS[2]


            box_size = gymapi.Vec3(box_x, box_y, box_z)
            box_sizes.append(box_size)
            box_size_tensor.append(torch.Tensor([box_x, box_y, box_z]).to(self.device).unsqueeze(0))
            asset_options = gymapi.AssetOptions()
            asset_options.density = 110
            box_asset = self.gym.create_box(self.sim, box_size.x, box_size.y, box_size.z, asset_options)


            box_idx = self.gym.find_asset_rigid_body_index(box_asset, "box")
            box_sensor_pose = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
            box_sensor_idx = self.gym.create_asset_force_sensor(box_asset, box_idx, box_sensor_pose)

            BOX_OFFSET = 0.01
            bottom_handle = self.gym.find_actor_rigid_body_handle(env_ptr, shelf_actor, "bottom")
            bottom_pose = self.gym.get_rigid_transform(env_ptr, bottom_handle)
            box_pose = gymapi.Transform()
            box_pose.p.x = bottom_pose.p.x -SHELF_DEPTH/2+SHELF_WALL_WIDTH/2 + box_size.x/2 + BOX_OFFSET
            box_pose.p.y = bottom_pose.p.y -SHELF_LENGTH/2+SHELF_WALL_WIDTH/2 + box_size.y/2 + BOX_OFFSET
            box_pose.p.z = bottom_pose.p.z + SHELF_WALL_WIDTH/2 + box_size.z/2 + BOX_OFFSET
            box_pose.r = gymapi.Quat(0, 0, 0, 1)

            
            box_handle = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 0, segmentationId=BOX_SEG_ID)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            sensor = self.gym.get_actor_force_sensor(env_ptr, box_handle, 0)

            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(env_ptr, box_handle, 0, gymapi.DOMAIN_SIM)
            box_idxs.append(box_idx)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            #create camera sensor
            # add camera sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = 128
            camera_props.height = 128
            camera_props.enable_tensors = True
            camera_position = gymapi.Vec3(.3, 0, shelf_pose.p.z - SHELF_LENGTH/2+.5)
            lookat_position = gymapi.Vec3(shelf_pose.p.x, shelf_pose.p.y, shelf_pose.p.z - SHELF_LENGTH/2+.2)
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            camera_idxs.append(camera_handle)
            self.gym.set_camera_location(camera_handle, env_ptr, camera_position, lookat_position)

            # obtain camera tensor
            segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
            depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            # wrap camera tensor in a pytorch tensor
            torch_seg_tensor = gymtorch.wrap_tensor(segmentation_tensor).unsqueeze(0)
            seg_tensors.append(torch_seg_tensor)
            torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor).unsqueeze(0)
            depth_tensors.append(torch_depth_tensor)
            # get camera matrix
            projection_matrix = torch.Tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle)).to(self.device)
            view_matrix = torch.Tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)).to(self.device)
            proj_matrixes.append(projection_matrix.unsqueeze(0))
            inv_view_matrix = torch.linalg.inv(view_matrix).to(self.device)
            inv_view_matrixes.append(inv_view_matrix.unsqueeze(0))

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.shelfs.append(shelf_actor)

        curr_proj_matrixes = torch.cat(proj_matrixes, 0)
        curr_inv_view_matrixes = torch.cat(inv_view_matrixes, 0)
        box_size_tensor = torch.cat(box_size_tensor, 0)
        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                                drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                                drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw
        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., -1] = 1
        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
            self.franka_grasp_pos, self.drawer_grasp_pos, self.franka_grasp_rot, self.drawer_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
                                     drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
                                     )

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target,
                                  self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        # reset props
        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.drawer_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, cabinet_dof_pos,
    franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
    # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

    # bonus if left finger is above the drawer handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                       torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    # reward for distance of each finger from the drawer
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                     torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # how far the cabinet has been opened out
    open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
        + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
        + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty

    # bonus for opening drawer properly
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    # prevent bad style in opening drawer
    rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                          torch.ones_like(rewards) * -1, rewards)
    
    # reset if drawer is open or max length reached
    reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
