import os
import random

import hydra
import numpy as np
import pytorch_lightning as zeus
import torch
from omegaconf import DictConfig, OmegaConf

from byoc.datasets import build_loader
from byoc.nnutils.byoc_trainer import BYOC_Registration


@hydra.main(config_name="config", config_path="byoc/configs")
def test(cfg: DictConfig) -> None:

    # --- Reproducibility | https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(cfg.system.random_seed)
    random.seed(cfg.system.random_seed)
    np.random.seed(cfg.system.random_seed)

    # --- load checkpoint ---
    ckpt_cfg = cfg.test.checkpoint
    if ckpt_cfg.name == "":
        # assume no checkpoint and run with untrained model
        exp_name = cfg.experiment.name
        exp_time = ckpt_cfg.time
        OmegaConf.set_struct(cfg, False)
        cfg.experiment.full_name = f"{exp_name}_{exp_time}"
        model = BYOC_Registration(cfg)
    else:
        exp_name = ckpt_cfg.name
        exp_time = ckpt_cfg.time
        ckpt_dir = os.path.join(
            cfg.paths.experiments_dir, f"{ckpt_cfg.name}_{ckpt_cfg.time}"
        )

        if ckpt_cfg.step == -1:
            ckpts = os.listdir(ckpt_dir)
            ckpts.sort()

            # pick last file by default -- most recent checkpoint
            ckpt_file = ckpts[-1]
            print(f"Using the last checkpoint: {ckpt_file}")
        else:
            epoch = ckpt_cfg.epoch
            step = ckpt_cfg.step
            ckpt_file = f"checkpoint-epoch={epoch:03d}-step={step:07d}.ckpt"

        checkpoint_path = os.path.join(ckpt_dir, ckpt_file)


        # model = BYOC_Registration.load_from_checkpoint(checkpoint_path, strict=False)
        model = BYOC_Registration(cfg)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    # get dataset split and get first item, useful when debugging
    loader = build_loader(cfg.dataset, split=cfg.test.split)
    loader.dataset.__getitem__(0)

    # update model with test configs
    assert model.model.geometric == True    # check that geometric registration is on
    model.model.augment_pc = False
    model.model.visual = False
    model.model.num_corres = cfg.test.align_cfg.num_correspodances
    model.model.align_cfg = cfg.test.align_cfg

    # -- test model --
    trainer = zeus.Trainer(gpus=1)
    trainer.test(model, loader, verbose=False)

if __name__ == "__main__":
    test()
