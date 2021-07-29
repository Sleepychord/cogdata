# -*- encoding: utf-8 -*-
# @File    :   zip_dataset.py
# @Time    :   2021/07/10 14:25:49
# @Author  :   Ming Ding
# @Contact :   dm18@mail.tsinghua.edu.cn

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
from cogdata.utils.logger import get_logger

from cogdata.utils.register import register


@register
class ZipDataset(Dataset):
    """Datasets for zip files
    """

    def __init__(self, path, *args, world_size=1, rank=0, transform_fn=None):
        """Split data for multiple processes. 
        Get the file pointer and filenames of valid samples.
        Set transform function.

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
        assert len(args) == 0, 'transform_fn is a kwarg.'
        self.path = path
        self.zip = zipfile.ZipFile(path)
        self.members = [
            info for info in self.zip.infolist()
            if info.filename[-1] != os.sep and '__MACOSX' not in info.filename
        ]
        # split by distributed
        if world_size > 1:
            self.members = self.members[rank::world_size]
        self.transform_fn = transform_fn
        self.zip.close()
        self.zip = None

    def __len__(self):
        """Get the total number of the valid samples.

        Returns
        -------
        int
            The total number of the valid samples.
        """
        return len(self.members)

    def __getitem__(self, idx):
        """Get a item by index

        Args:
            idx(int):The selected item's index.

        Returns:
            if ``transform_fn`` is not ``None``, 
                return a result of the ``transform_fn``.
            if ``transform_fn`` is ``None``, 
                return a tuple containing

                - fp(file pointer): file pointer fo zip file.
                - full_filename(str): filename of the image.
                - file_size: The size of a raw file in the zip file.
        """
        if self.zip is None:
            self.zip = zipfile.ZipFile(self.path)
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
        """Delete the Dataset. Close the zip files"""
        self.zip.close()
