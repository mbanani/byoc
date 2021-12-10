"""
Create a 3DMatch Pointcloud registration dataset for the evaluation set. Note that this
is only the test set; we don't use the benchmark's training set.
"""
import argparse
import os
import pickle

import numpy as np


def read_log_file(path):
    """
    Read the log file for 3D Match's log files
    """
    with open(path, "r") as f:
        log_lines = f.readlines()
        log_lines = [line.strip() for line in log_lines]

    num_logs = len(log_lines) // 5
    transforms = []

    for i in range(0, num_logs, 5):
        meta_data = np.fromstring(log_lines[i], dtype=int, sep=" \t")
        transform = np.zeros((4, 4), dtype=float)
        for j in range(4):
            transform[j] = np.fromstring(log_lines[i + j + 1], dtype=float, sep=" \t")
        transforms.append((meta_data, transform))

    return transforms


def create_3dmatch_dict(data_root, dict_path):
    class_id = "3DMatch_PCReg"
    data_dict = {}

    data_dict = {class_id: {}}

    # get the scene names
    scenes = os.listdir(data_root)
    scenes = [scene for scene in scenes if "-evaluation" not in scene]

    # read the log file for each scene
    data_dict = {}
    for scene in scenes:
        # get the log files
        log_path = os.path.join(data_root, f"{scene}-evaluation/gt.log")
        data_dict[scene] = {"transforms": read_log_file(log_path)}

        # get the ply files
        pc_files = os.listdir(os.path.join(data_root, scene))
        pc_files = [pc for pc in pc_files if "ply" in pc]
        data_dict[scene]["pointclouds"] = {}
        for pc_file in pc_files:
            index = pc_file.split(".ply")[0].split("cloud_bin_")[1]
            index = int(index)
            data_dict[scene]["pointclouds"][index] = f"{scene}/{pc_file}"

    # save dictionary as pickle in output path
    with open(dict_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("dict_path", type=str)
    args = parser.parse_args()

    create_3dmatch_dict(args.data_root, args.dict_path)
