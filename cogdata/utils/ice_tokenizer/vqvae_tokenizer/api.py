import importlib
import torch
import json
import math
import os
from datetime import datetime
from tqdm import tqdm
import time
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import torch.nn.functional as F

PRETRAINED_MODEL_FILE = ""

def new_module(config):
    '''in config:
            "target": module type
            "params": dict of params'''
    if type(config) == str:
        with open(config, 'r') as file:
            config = json.load(file)
    assert type(config) == dict
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config.get('target').rsplit(".", 1)
    model = getattr(importlib.import_module(module, package=None), cls)(**config.get("params", dict()))
    
    return model

def load_ckpt(model, path=None):
    if path is None:
        path = PRETRAINED_MODEL_FILE
    sd = torch.load(path, map_location="cpu")['module']
    model.load_state_dict(sd, strict=False)
    return model

def load_default_HVQVAE():
    config = {
        "target": "vqvae.HVQVAE",
        "params": {
            "levels": 3,
            "embedding_dim": 256,
            "codebook_scale": 1,
            "down_sampler_configs": [
                {
                    "target": "vqvae.ResidualDownSample",
                    "params": {
                        "in_channels": 256
                    }
                },
                {
                    "target": "vqvae.ResidualDownSample",
                    "params": {
                        "in_channels": 256
                    }
                }
            ],
            "enc_config": {
                    "target": "vqvae.Encoder",
                    "params": {
                        "num_res_blocks": 2,
                        "channels_mult": [1,2,4]
                    }
            },
            "quantize_config": {
                    "target": "vqvae.VectorQuantizeEMA",
                    "params": {
                        "hidden_dim": 256,
                        "embedding_dim": 256,
                        "n_embed": 20000,
                        "training_loc": false
                    }
            },
            "dec_configs": [
                {
                    "target": "vqvae.Decoder",
                    "params": {
                        "channels_mult": [1,1,1,2,4]
                    }
                },
                {
                    "target": "vqvae.Decoder",
                    "params": {
                        "channels_mult": [1,1,2,4]
                    }
                },
                {
                    "target": "vqvae.Decoder",
                    "params": {
                        "channels_mult": [1,2,4]
                    }
                }
            ]
        }
    }
    return new_module(config)


if __name__ == '__main__':
    pass