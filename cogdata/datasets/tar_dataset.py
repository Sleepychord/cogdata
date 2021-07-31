# -*- encoding: utf-8 -*-
# @File    :   tar_dataset.py
# @Time    :   2021/07/10 23:31:21
# @Author  :   Ming Ding 
# @Contact :   dm18@mail.tsinghua.edu.cn

# here put the import lib
import os
import sys
import math
import random
from torch.utils.data import Dataset
import tarfile
import torch.distributed as dist

from cogdata.utils.register import register


@register
class TarDataset(Dataset):
    def __init__(self, path, world_size=1, rank=0, transform_fn=None):
        """Split data for multiple process, Get the file pointer and filenames of valid samples, set transform function.

        Parameters
        ----------
        path:str
            The path of the zip file.
        world_size:int
            The total number of GPUs
        rank:int
            The local rank of current process
        transform_fn:function
            Used in __getitem__
        """
        self.path = path
        self.tar = tarfile.TarFile(path)
        self.members = [
            x for x in self.tar.getmembers()
            if x.isfile() and '__MACOSX' not in x.name
        ]
        # split by distributed
        if world_size > 1:
            self.members = self.members[rank::world_size]

        self.transform_fn = transform_fn
        self.tar.close()
        self.tar = None

    def __len__(self):
        """Get the total number of the valid samples."""
        return len(self.members)

    def __getitem__(self, idx):
        """Get a item by index

        Parameters
        ----------
        idx:int
            The selected item's index.

        Returns
        -------
        item:Tensor
            A torch tensor built from numpy array
        """
        if self.tar is None:
            self.tar = tarfile.TarFile(self.path)
        target_info = self.members[idx]
        fp = self.tar.extractfile(target_info)
        full_filename = self.members[idx].name
        file_size = self.members[idx].size
        if self.transform_fn is not None:
            return self.transform_fn(fp, full_filename, file_size)
        else:
            return fp, full_filename, file_size

    def __del__(self):
        if self.tar is not None:
            self.tar.close()