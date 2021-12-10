import os
import socket
import pickle

import pytorch3d
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .pointcloud_dataset import PointcloudDataset
from .video_dataset import VideoDataset

# Define some important paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# get dataset root
RGBD_3D_ROOT = ""
GEO_3DMATCH_ROOT = ""
SCANNET_ROOT = ""


def build_dataset(cfg, split, overfit=None):
    """
    Builds a dataset from the provided dataset configs.
    Configs can be seen is configs/config.py
    """
    if cfg.name == "ScanNet":
        root_path = SCANNET_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"data/scannet_{split}.pkl")
        data_dict = load_pickle(dict_path)
        dataset = VideoDataset(cfg, root_path, data_dict, split)

        # Reduce ScanNet validation size to allow for more frequent validation
        if split == "valid":
            dataset.instances = dataset.instances[::10]
    elif cfg.name == "3DMatch":
        root_path = RGBD_3D_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"data/3dmatch_{split}.pkl")
        data_dict = load_pickle(dict_path)
        dataset = VideoDataset(cfg, root_path, data_dict, split)

        # Reduce ScanNet validation size to allow for more frequent validation
        if split == "valid":
            dataset.instances = dataset.instances[::10]
    elif cfg.name == "3DMatch_PCReg":
        assert split == "test"
        root_path = GEO_3DMATCH_ROOT
        dict_path = os.path.join(PROJECT_ROOT, "data/3dmatch_reg_test.pkl")
        data_dict = load_pickle(dict_path)
        dataset = PointcloudDataset(cfg, root_path, data_dict, split, rotated=False)
    else:
        raise ValueError("Dataset name {} not recognized.".format(cfg.name))

    # Overfit only loads a single batch for easy debugging/sanity checks
    if overfit is not None:
        assert type(overfit) is int
        dataset.instances = dataset.instances[: cfg.batch_size] * overfit

    return dataset


def build_loader(cfg, split, overfit=None):
    """
    Builds the dataset loader (including getting the dataset).
    """
    dataset = build_dataset(cfg, split, overfit)
    shuffle = (split == "train") and (not overfit)
    batch_size = cfg.batch_size

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=custom_collate,
        num_workers=cfg.num_workers,
    )

    return loader


def custom_collate(data):
    out = {}
    keys = data[0].keys()

    for key in keys:
        if "points_rgb_" in key:
            view_i = key.split("_")[-1]
            xyz = [ins[f"points_{view_i}"] for ins in data]
            rgb = [ins[f"points_rgb_{view_i}"] for ins in data]
            out[key] = pytorch3d.structures.Pointclouds(xyz, None, rgb)
        elif "points_" in key:
            out[key] = pytorch3d.structures.Pointclouds([ins[key] for ins in data])
        else:
            out[key] = default_collate([ins[key] for ins in data])

    return out


# load pickle
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
