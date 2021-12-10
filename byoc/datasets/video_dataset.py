import numpy as np
import torch
from torchvision import transforms as transforms

from ..utils.util import get_grid, grid_to_pointcloud
from .abstract import AbstractDataset


class VideoDataset(AbstractDataset):
    def __init__(self, cfg, root_path, data_dict, split):
        name = cfg.name
        super(VideoDataset, self).__init__(name, split, root_path)
        self.cfg = cfg
        self.split = split
        self.num_views = cfg.num_views
        self.view_spacing = cfg.view_spacing
        self.image_dim = cfg.img_dim

        self.data_dict = data_dict
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.image_dim),
                transforms.CenterCrop(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
            ]
        )

        # The option to do strided frame pairs is to under sample Validation and Test
        # sets since there's a huge number of frames to start with. Since RGBD is much
        # smaller, we use the non-strided version. An example of strided vs non strided
        # for a view spacing of 10 and frame pairs
        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        self.instances = self.dict_to_instances(self.data_dict, strided)
        self.grid = get_grid(self.image_dim, self.image_dim)

        # Print out dataset stats
        print("================================")
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def dep_transform(self, dep):
        h, w = dep.shape
        left = int(round((w - h) / 2.0))
        right = left + h
        dep = dep[:, left:right]
        dep = torch.Tensor(dep[None, None, :, :]).float()
        dep = torch.nn.functional.interpolate(dep, (self.image_dim, self.image_dim))[0]
        return dep

    def __getitem__(self, index):
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # Read in separate instances
        P_rel = s_instance[f_ids[0]]["extrinsic"]

        for i, id_i in enumerate(f_ids):
            rgb = self.get_rgb(s_instance[id_i]["rgb_path"])
            smaller_dim = rgb.height
            crop_offset = (rgb.width - rgb.height) / 2

            # -- Transform K to handle image resize and crop
            K = s_instance[f_ids[0]]["intrinsic"][:3, :3].copy()
            K[0, 2] -= crop_offset  # handle cropped width
            K[:2, :] *= self.image_dim / smaller_dim  # handle resizing
            K = torch.tensor(K).float()
            output["K"] = K

            # transform and save rgb
            rgb = self.rgb_transform(rgb)
            output[f"rgb_{i}"] = rgb

            # Resize depth and scale to meters according to ScanNet Docs
            # http://kaldir.vc.in.tum.de/scannet_benchmark/documentation
            dep = self.get_img(s_instance[id_i]["dep_path"])
            dep = self.dep_transform(dep)
            dep = dep / 1000.0
            output[f"depth_{i}"] = dep

            # generate pointcloud
            pc = grid_to_pointcloud(K.inverse(), dep, self.grid)
            pc_valid = pc[pc[:, 2] != 0]
            pc_rgb = (rgb * 0.5 + 0.5).view(3, -1).transpose(0, 1)
            pc_rgb = pc_rgb[pc[:, 2] != 0]

            output[f"points_{i}"] = pc_valid
            output[f"points_rgb_{i}"] = pc_rgb

            # Some rotation conversions -- absolute -> relative reference
            P = s_instance[id_i]["extrinsic"]
            P = torch.tensor(np.linalg.inv(P) @ P_rel).float()
            output[f"Rt_{i}"] = P[:3, :]

            # Set identities for xyz and quat
            output[f"path_{i}"] = s_instance[id_i]["rgb_path"]

        return output

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        for cls_id in data_dict:
            for s_id in data_dict[cls_id]:
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()

                if strided:
                    frames = frames[:: self.view_spacing]
                    stride = 1
                else:
                    stride = self.view_spacing

                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    f_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])

                    # Hacky way of getting source to be in the middle for triplets
                    mid = self.num_views // 2
                    f_ids = f_ids[mid:] + f_ids[:mid]
                    instances.append([cls_id, s_id, tuple(f_ids)])

        return instances
