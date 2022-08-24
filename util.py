


#!/home/aurmr/anaconda3/envs/rlgpu3/bin/python

from turtle import end_poly
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
import numpy as np
import torch
from torch.nn.functional import grid_sample
from torch.nn import ReplicationPad2d
import random
import time
from torchvision.utils import save_image
from pytorch3d.common.workaround import symeig3x3
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


#If seg_map is not none, crop around the segmap mask
def backproject_tensor(proj_mat, inv_view_mat, depth_image, seg_map=None, crop_size=64):

    xs = torch.linspace(0, depth_image.shape[2]-1, depth_image.shape[2])
    ys = torch.linspace(0, depth_image.shape[1]-1, depth_image.shape[1])
    (v, u) = torch.meshgrid(ys, xs)

    u = u.unsqueeze(0).expand(depth_image.shape[0], -1, -1).to(depth_image.device)
    v = v.unsqueeze(0).expand(depth_image.shape[0], -1, -1).to(depth_image.device)

    cam_height = depth_image.shape[1]
    cam_width = depth_image.shape[2]

    if seg_map is not None:
        centroid_x = (torch.sum(u*seg_map, (1,2))/torch.count_nonzero(u*seg_map, (1,2))).to(torch.int).unsqueeze(-1).to(torch.float)
        centroid_y = (torch.sum(v*seg_map, (1,2))/torch.count_nonzero(v*seg_map, (1,2))).to(torch.int).unsqueeze(-1).to(torch.float)
        grid_x = batch_tensor_linspace(centroid_x - crop_size//2, centroid_x + crop_size//2-1, steps=crop_size)
        grid_y = batch_tensor_linspace(centroid_y - crop_size//2, centroid_y + crop_size//2-1, steps=crop_size)
        grid = torch.cat((grid_x.expand(-1, crop_size, -1).unsqueeze(-1), grid_y.transpose(1,2).expand(-1, -1, crop_size).unsqueeze(-1)), dim=-1)
        
        grid[..., 0] = (grid[..., 0].to(torch.float) - cam_width//2)/(cam_width//2)
        grid[..., 1] = (grid[..., 1].to(torch.float) - cam_height//2)/(cam_height//2)
        # box_left = centroid_x - crop_size//2
        # box_top = centroid_y - crop_size//2

        # pre = seg_map[0].cpu().to(torch.float)*255
        # pre[(centroid_x[0][0].to(torch.int)-64):(centroid_x[0][0].to(torch.int)+64), (centroid_y[0][0].to(torch.int)-64):(centroid_y[0][0].to(torch.int)+64)] = .5 + \
        #     pre[(centroid_x[0][0].to(torch.int)-64):(centroid_x[0][0].to(torch.int)+64), (centroid_y[0][0].to(torch.int)-64):(centroid_y[0][0].to(torch.int)+64)]/2
        # save_image(pre, "/tmp/pre.bmp")
        depth_image = grid_sample(depth_image.unsqueeze(1), grid).squeeze()
        seg_map = grid_sample(seg_map.unsqueeze(1).to(torch.float), grid).squeeze().to(torch.int)
        # save_image(seg_map[0].cpu().to(torch.float)*255, "/tmp/post.bmp")
        u = grid_sample(u.unsqueeze(1).to(torch.float), grid).squeeze().to(torch.int)
        v = grid_sample(v.unsqueeze(1).to(torch.float), grid).squeeze().to(torch.int)
    
    # u_tmp = u.clone()
    # u = v
    # v = u_tmp

    fx = proj_mat[:, 0, 0].unsqueeze(1).unsqueeze(2)
    fy = proj_mat[:, 1, 1].unsqueeze(1).unsqueeze(2)

    u = -(2*u-cam_width)/cam_width
    v = (2*v-cam_height)/cam_height
    x_e = ((u) * depth_image) / fx
    y_e = ((v) * depth_image) / fy
   
    xyz_img = torch.stack([x_e, y_e, depth_image, torch.ones(depth_image.shape).to(depth_image.device)], dim=1)
    xyz_img = xyz_img.flatten(2,3).swapaxes(1,2)
  
    pointcloud = torch.bmm(xyz_img, inv_view_mat)
    pointcloud = pointcloud[:, :, 0:3]/(pointcloud[:, :, 3].unsqueeze(-1).expand((-1, -1, 3)))

    organized_pc = pointcloud[:, :, 0:3].view(seg_map.shape[0], seg_map.shape[1], seg_map.shape[2], 3)

    return (pointcloud[:, :, 0:3], seg_map, organized_pc)


def batch_cov(points, sample_size=9):
    size = points.size()
    B = int(torch.prod(torch.Tensor(list(size[:-2]))))
    D = int(size[-1])
    N = int(size[-2])
    final_size = list(size)
    # final_size.append(N)
    final_size.append(D)
    mean = torch.mean(points, dim=-2).unsqueeze(-2)
    diffs = (points - mean).reshape(B*N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(*(final_size))
    bcov = prods.sum(dim=-3) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def estimate_pc_normals(organized_points, kernel_size=3):
    width = organized_points.shape[1]
    height = organized_points.shape[2]

    xs = torch.arange(0, width, 1).to(organized_points.device)
    ys = torch.arange(0, height, 1).to(organized_points.device)
    grid = torch.stack(torch.meshgrid((xs, ys)), dim=-1).to(organized_points.device)
    grid_offset = grid + kernel_size-1
    grid_slices = batch_tensor_linspace(grid.to(torch.float), grid_offset.to(torch.float), kernel_size).to(torch.long).unsqueeze(-1).expand(-1, -1, -1, -1, kernel_size).to(organized_points.device).clone()
    grid_slices[:, :, 1, :, :] = grid_slices[:, :, 1, :, :].transpose(-2, -1)
    pc_mean = torch.mean(organized_points, dim=(1,2)).unsqueeze(1).unsqueeze(1)
    points_centered = organized_points - pc_mean 
    pad = ReplicationPad2d(kernel_size//2).to(organized_points.device)
    org_pc_padded = pad(points_centered.permute((0, 3, 2, 1))).permute((0, 3, 2, 1))
    point_neighborhoods = org_pc_padded[:, grid_slices[:, :, 0, :, :].flatten(), grid_slices[:, :, 1, :, :].flatten(), :]
    point_neighborhoods = point_neighborhoods.reshape(-1, width, height, kernel_size*kernel_size, 3)

    point_neighborhoods = point_neighborhoods*1000
    cov = batch_cov(point_neighborhoods)
    curvatures, local_coord_frames = symeig3x3(cov, eigenvectors=True)
    normals = local_coord_frames[..., 0]
    return normals.view(normals.shape[0], -1, 3), local_coord_frames.view(normals.shape[0], -1, 3, 3)

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
    
    expanded_dims = ([-1]*start.dim())
    expanded_dims.append(steps)
    start = start.unsqueeze(-1).expand(*expanded_dims)
    end = end.unsqueeze(-1).expand(*expanded_dims)

    out = start_w * start + end_w * end

    return out

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)