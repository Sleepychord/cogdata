# -*- encoding: utf-8 -*-
'''
@File    :   zip_dataset.py
@Time    :   2021/07/10 14:25:49
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
import ctypes
import io

import numpy as np
import torch
import torch.nn.functional as F
import zipfile
import PIL
import torch.distributed as dist

from torch.utils.data import Dataset, IterableDataset
from PIL import Image
from ..utils.logger import get_logger

class ZipDataset(Dataset):
    def __init__(self, path, transform_fn=None):
        self.zip = zipfile.ZipFile(path)
        self.members = [
            info for info in self.zip.infolist() 
                if info.filename[-1] != os.sep and '__MACOSX' not in info.filename
        ]
        # split by distributed
        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()  
            self.members = [
                x for i, x in enumerate(self.members) 
                if i % num_replicas == rank
            ]
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        target_info = self.members[idx]
        full_filename = self.members[idx].filename
        file_size = self.members[idx].file_size
        try:
            fp = self.zip.open(target_info)
        except zipfile.BadZipFile as e:
            fp = None
            get_logger().warning(f'{full_filename} is a bad zipfile.')

        if self.transform_fn is not None:
            return self.transform_fn(fp, full_filename, file_size)
        else:
            return fp, full_filename, file_size
    
    def __del__(self):
        self.zip.close()