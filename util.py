


#!/home/aurmr/anaconda3/envs/rlgpu3/bin/python

from turtle import end_poly
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
import numpy as np
import torch
import random
import time

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[:, 0]
    q1 = Q[:, 1]
    q2 = Q[:, 2]
    q3 = Q[:, 3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = torch.stack([torch.stack([r00, r01, r02], -1),
                           torch.stack([r10, r11, r12], -1),
                           torch.stack([r20, r21, r22], -1)], -1)
    # print(rot_matrix.shquaternion_to_matrixape)
    return rot_matrix       

def backproject_tensor(proj_mat, inv_view_mat, depth_image, seg_map):

    xs = torch.linspace(0, depth_image.shape[1]-1, depth_image.shape[1])
    ys = torch.linspace(0, depth_image.shape[2]-1, depth_image.shape[2])
    (u, v) = torch.meshgrid(xs, ys)
    u = u.unsqueeze(0).expand(depth_image.shape[0], -1, -1).to(depth_image.device)
    v = v.unsqueeze(0).expand(depth_image.shape[0], -1, -1).to(depth_image.device)

    cam_height = depth_image.shape[2]
    cam_width = depth_image.shape[1]

    fx = proj_mat[:, 0, 0].unsqueeze(1).unsqueeze(2)
    fy = proj_mat[:, 1, 1].unsqueeze(1).unsqueeze(2)

    u = (2*u-cam_width)/cam_width
    v = -(2*v-cam_height)/cam_height
    x_e = ((u) * depth_image) / fx
    y_e = ((v) * depth_image) / fy
   
    xyz_img = torch.stack([y_e, x_e, depth_image, torch.ones(depth_image.shape).to(depth_image.device)], dim=1)
    xyz_img = xyz_img.flatten(2,3).swapaxes(1,2)
  
    pointcloud = torch.bmm(xyz_img, inv_view_mat)
    pointcloud = pointcloud[:, :, 0:3]/(pointcloud[:, :, 3].unsqueeze(-1).expand((-1, -1, 3)))

    return pointcloud[:, :, 0:3]



def match_planes(normals, plane_points, points):

    MIN_DIST = .01

    #expand tensors into the correct shape
    points = points.unsqueeze(1).expand(-1, plane_points.shape[1], -1, -1)
    plane_points = plane_points.unsqueeze(2).expand(-1, -1, points.shape[2], -1)
    normals = normals.unsqueeze(2).expand(-1, -1, points.shape[2], -1)
    
    v = points-plane_points
    dists = torch.abs(torch.bmm(v.contiguous().view(-1, 1, 3), normals.contiguous().view(-1, 3, 1))).reshape(v.shape[0:-1])
    # matching_planes = torch.argmin(dists, 1)
    matching_planes = dists < MIN_DIST
    return matching_planes

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

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)