from typing import List, Optional

import os
import shutil
import MinkowskiEngine as ME
import pytorch3d
import torch
from MinkowskiEngine.utils.quantization import sparse_quantize
from torch import nn as nn


def nn_gather(points, indices):
    # expand indices to same dimensions as points
    indices = indices[:, :, None]
    indices = indices.expand(indices.shape[0], indices.shape[1], points.shape[2])
    return points.gather(1, indices)


def pointclouds_to_minkowski(pointcloud, voxel_size):
    points = pointcloud.points_list()
    pc_q = [(pc / voxel_size).float() for pc in points]
    pc_q = [sparse_quantize(pc.cpu(), return_index=True) for pc in pc_q]
    mink = ME.utils.batched_coordinates([pc[0] for pc in pc_q])
    pc_X = [points[i][pc_q[i][1]] for i in range(len(points))]
    return mink, pc_X


def minkowski_to_pointclouds(mink_pc, sparse_X):
    pcs_F = mink_pc.decomposed_features
    pcs_F = [nn.functional.normalize(pc, dim=1) for pc in pcs_F]
    points = pytorch3d.structures.Pointclouds(points=sparse_X, features=pcs_F)
    return points


def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


@torch.jit.script
def grid_to_pointcloud(K_inv, depth, grid: Optional[torch.Tensor]):
    _, H, W = depth.shape

    if grid is None:
        grid = get_grid(H, W)

    # Apply inverse projection
    points = depth * grid

    # Invert intriniscs
    points = points.view(3, H * W)
    points = K_inv @ points
    points = points.permute(1, 0)

    return points

# make directory
def makedir(path, replace_existing=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif replace_existing:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        print("Beware .. path {} already exists".format(path))

def full_detach(x):
    x_type = type(x)
    if x_type == dict:
        return detach_dictionary(x)
    elif x_type == tuple:
        return tuple([full_detach(x_el) for x_el in x])
    elif x_type == list:
        return [full_detach(x_el) for x_el in x]
    elif x_type == torch.Tensor:
        return x.detach().cpu()
    elif x_type == pytorch3d.structures.pointclouds.Pointclouds:
        return x.detach().cpu()
    else:
        raise ValueError(f"Unable to detach input of type {x_type}")


def detach_dictionary(gpu_dict):
    for k in gpu_dict:
        old_val = gpu_dict[k]
        gpu_dict[k] = full_detach(old_val)

    return gpu_dict


