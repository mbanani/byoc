import MinkowskiEngine as ME
import pytorch3d
import torch
from torch import nn as nn
from torch.nn.functional import grid_sample

from ..utils.transformations import random_Rt, transform_points_Rt
from ..utils.util import minkowski_to_pointclouds, nn_gather, pointclouds_to_minkowski
from .alignment import align
from .backbones import ResNetEncoder
from .correspondence import get_correspondences, transfer_correspondances
from .minkunet import MinkUNet34C


class BYOC(nn.Module):
    def __init__(self, cfg):
        super(BYOC, self).__init__()

        # Define model parameters
        chan_in = 3
        feat_dim = cfg.feat_dim
        self.voxel_size = cfg.voxel_size

        self.visual = cfg.visual_registration
        self.geometric = cfg.geometric_registration

        assert self.visual or self.geometric, "Need to do at least one"

        if self.visual:
            self.image_encoder = ResNetEncoder(chan_in, feat_dim)

        if self.geometric:
            self.point_encoder = MinkUNet34C(chan_in, feat_dim)

        self.num_corres = cfg.alignment.num_correspodances
        self.align_cfg = cfg.alignment
        self.v2g_cfg = cfg.v2g_transfer
        self.augment_pc = cfg.Rt_augmentation

        self.mlp_project = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            # It's actually BatchNorm given that is Num_Points x Features
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, rgbs, pcs, K):
        # Encode visual and geometric features
        if self.geometric:
            geo_pc_0 = self.encode_pc(pcs[0], augment=False)
            # only augment during training
            augment = self.augment_pc and (not self.eval())
            geo_pc_1 = self.encode_pc(pcs[1], augment=augment)
        else:
            # used for visual encoding
            geo_pc_0 = pcs[0]
            geo_pc_1 = pcs[1]

        if self.visual:
            vis_pc_0 = self.encode_image(rgbs[0], K, geo_pc_0)
            vis_pc_1 = self.encode_image(rgbs[1], K, geo_pc_1)

        # note that both xyz are the same!
        xyz_0 = geo_pc_0.points_padded()
        xyz_1 = geo_pc_1.points_padded()

        output = {}
        # ===== Visual Alignment =====
        if self.visual:
            vis_corr = get_correspondences(vis_pc_0, vis_pc_1, self.num_corres)
            Rt_vis, vis_loss, vis_residual = align(
                vis_corr, xyz_0, xyz_1, self.align_cfg
            )
            vis_corr_output = [
                nn_gather(xyz_0, vis_corr[0]),
                nn_gather(xyz_1, vis_corr[1]),
                vis_corr[2],
            ]

            output["vis_corr_loss"] = vis_loss
            output["vis_corr"] = vis_corr_output
            output["Rt_vis"] = Rt_vis
            output["vis_pc_0"] = vis_pc_0
            output["vis_pc_1"] = vis_pc_1

            # calculate variances
            std_vis = [pc.std(dim=0).mean() for pc in vis_pc_0.features_list()]
            std_vis = torch.tensor(std_vis)
            output["feat_std_vis"] = std_vis

        # ===== Geometric Alignment =====
        if self.geometric:
            geo_corr = get_correspondences(geo_pc_0, geo_pc_1, self.num_corres)
            Rt_geo, geo_loss, _ = align(
                geo_corr, xyz_0, xyz_1, self.align_cfg, overwrite_weighting="none"
            )

            geo_corr_output = [
                nn_gather(xyz_0, geo_corr[0]),
                nn_gather(xyz_1, geo_corr[1]),
                geo_corr[2],
            ]

            output["geo_corr_loss"] = geo_loss
            output["geo_corr"] = geo_corr_output
            output["Rt_geo"] = Rt_geo
            output["geo_pc_0"] = geo_pc_0
            output["geo_pc_1"] = geo_pc_1

            # calculate variances
            std_geo = [pc.std(dim=0).mean() for pc in geo_pc_0.features_list()]
            std_geo = torch.tensor(std_geo)
            output["feat_std_geo"] = std_geo

        # ===== Visual -> Geometric Transfer =====
        if self.visual and self.geometric:
            v2g_loss, v2g_weight = transfer_correspondances(
                vis_corr,
                geo_pc_0,
                geo_pc_1,
                self.v2g_cfg,
                self.mlp_project,
                vis_residual,
            )
            v2g_corr_output = [vis_corr_output[0], vis_corr_output[1], v2g_weight]

            output["v2g_corr_loss"] = v2g_loss
            output["v2g_corr"] = v2g_corr_output
            output["Rt_v2g"] = Rt_vis

        return output

    def encode_image(self, rgb, K, pc):
        batch_size, _, H, W = rgb.shape
        assert H == W

        feats = self.image_encoder(rgb)
        feats = nn.functional.normalize(feats, dim=1)

        # Convert XYZ to pixel coordinates
        pcX = pc.points_padded()
        uvd = pcX.bmm(K.transpose(1, 2))
        uv = uvd[:, :, :2] / uvd[:, :, 2:3].clamp(min=1e-9)
        uv = (2 * uv / H) - 1.0

        # sample points
        num_points = pc.num_points_per_cloud()
        pc_F = grid_sample(feats, uv.unsqueeze(1), align_corners=False)
        pc_F = pc_F[:, :, 0].transpose(1, 2)
        pc_F = [pc_F[i, 0 : num_points[i]] for i in range(batch_size)]

        points = pytorch3d.structures.Pointclouds(pc.points_list(), features=pc_F)

        return points

    def encode_pc(self, points, augment=False):
        # Augment pointcloud
        if augment:
            batch_size = len(points)
            num_points = points.num_points_per_cloud()

            aug_Rt = random_Rt(batch_size, max_R=0.7, max_t=0.5, device=points.device)
            pc_X = points.points_padded()
            pc_X = transform_points_Rt(pc_X, aug_Rt)
            pc_X = [pc_X[i, 0 : num_points[i]] for i in range(batch_size)]
            points = pytorch3d.structures.Pointclouds(pc_X)

        # compute features using augmented pointcloud
        quant_pc, sparse_X = pointclouds_to_minkowski(points, self.voxel_size)
        quant_pc = quant_pc.to(points.device)
        fake_fts = torch.ones_like(quant_pc[:, 1:]).float()
        quant_pc = ME.SparseTensor(fake_fts, quant_pc)
        quant_pc = self.point_encoder(quant_pc)
        points = minkowski_to_pointclouds(quant_pc, sparse_X)

        # reverse augmentation to take points back to original location.
        if augment:
            num_points = points.num_points_per_cloud()
            pc_X = points.points_padded()
            pc_F = points.features_list()
            pc_X = transform_points_Rt(pc_X, aug_Rt, inverse=True)
            pc_X = [pc_X[i, 0 : num_points[i]] for i in range(batch_size)]
            points = pytorch3d.structures.Pointclouds(pc_X, features=pc_F)

        return points
