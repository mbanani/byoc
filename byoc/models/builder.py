import torch

from .byoc import BYOC

def build_model(cfg):
    if cfg.name == "BYOC":
        model = BYOC(cfg)
    else:
        raise ValueError("Model {} is not recognized.".format(cfg.name))

    return model
