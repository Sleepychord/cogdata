# -*- encoding: utf-8 -*-
'''
@File    :   tar_dataset.py
@Time    :   2021/07/10 23:31:21
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from torch.utils.data import Dataset
import tarfile
import torch.distributed as dist

class TarDataset(Dataset):
    def __init__(self, path, transform_fn=None):
        self.tar = tarfile.TarFile(path)
        self.members = [
            x for x in self.tar.getmembers()
            if x.isfile() and '__MACOSX' not in x.name
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
        fp = self.tar.extractfile(target_info)
        full_filename = self.members[idx].name
        file_size = self.members[idx].size
        if self.transform_fn is not None:
            return self.transform_fn(fp, full_filename, file_size)
        else:
            return fp, full_filename