import numpy as np
import pytorch3d
import torch

from .abstract import AbstractDataset


class PointcloudDataset(AbstractDataset):
    def __init__(self, cfg, root_path, data_dict, split, rotated=False):
        name = cfg.name
        super(PointcloudDataset, self).__init__(name, split, root_path)
        self.cfg = cfg
        self.split = split

        self.data_dict = data_dict
        self.rotated = rotated

        self.instances = self.dict_to_instances(self.data_dict)
        # Print out dataset stats
        print("================================")
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def getraw(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        cls_id, s_id, f_ids, transform = self.instances[index]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # load pointclouds
        for i in range(2):
            pc_path = self.data_dict[s_id]["pointclouds"][f_ids[i]]

            pc = self.get_pointcloud(pc_path).points_list()[0]
            output[f"points_{i}"] = pc

        # get transform
        output["Rt_0"] = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        Rt = np.linalg.inv(transform)[:3, :]
        R = Rt[:, :3] / (Rt[:, :3] ** 2).sum(axis=1)[:, None]
        output["Rt_1"] = torch.tensor(np.concatenate((R, Rt[:, 3:4]), axis=1)).float()

        if self.rotated:
            pc_0 = output["points_0"]
            pc_1 = output["points_1"]
            Rt_1 = output["Rt_1"]

            # genereate two random rotation matrices
            R_rand_0 = pytorch3d.transforms.random_rotation().float()
            R_rand_1 = pytorch3d.transforms.random_rotation().float()

            # transpose r to handle the fact that P in num_points x 3
            # yT = (RX)T = XT @ RT
            # rotate pc_0 and pc_1 with R_rand_0
            pc_0 = pc_0 @ R_rand_0.transpose(0, 1)
            pc_1 = pc_1 @ R_rand_0.transpose(0, 1)

            # rotate pc_1 and Rt_1 with R_rand_1
            pc_1 = pc_1 @ R_rand_1.transpose(0, 1)

            R = Rt_1[:3, :3]
            t = Rt_1[:3, 3:4]

            """
            # calculating augment Rt is a bit tricky
            Y = RX + t
            X' = R0 @ X -> X = R0^-1 X'
            Y' = R1 @ R0 @ Y -> Y = R0-1 @ R1-1 @ Y'

            R0-1 R1-1 Y' = R(R0-1 X') + t
            Y- = R1 R0 R R0^-1 X' + R1 R0 t
            """
            R = R_rand_1 @ R_rand_0 @ R @ R_rand_0.transpose(0, 1)
            t = R_rand_1 @ R_rand_0 @ t
            Rt_1 = torch.cat((R, t), dim=1)

            # reasign output
            output["points_0"] = pc_0
            output["points_1"] = pc_1
            output["Rt_1"] = Rt_1

        return output

    def dict_to_instances(self, data_dict):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        cls_id = "3DMatch_PCReg"
        for s_id in data_dict:
            # get pairs from transformations
            transforms = data_dict[s_id]["transforms"]

            for pair in transforms:
                meta, transform = pair
                # Hacky way of getting source to be in the middle for triplets
                instances.append([cls_id, s_id, (meta[0], meta[1]), transform])

        return instances
