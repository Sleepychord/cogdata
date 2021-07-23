# -*- encoding: utf-8 -*-
'''
@File    :   binary_dataset.py
@Time    :   2021/07/10 23:39:23
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from cogdata.utils.register import register

@register
class BinaryDataset(Dataset):
    def __init__(self, path, length_per_sample, dtype='int32', preload=False, **kwargs):
        self.length_per_sample = length_per_sample
        self.dtype = np.dtype(dtype)
        if preload:
            self.bin = np.fromfile(path, dtype=self.dtype).reshape(-1, length_per_sample)
        else:
            with open(path, 'r') as fid:
                nbytes = fid.seek(0, 2)
                flen = fid.tell() // self.dtype.itemsize
            self.bin = np.memmap(path, dtype=self.dtype, shape=(flen // length_per_sample, length_per_sample))
    
    def __len__(self):
        return self.bin.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(self.bin[index])
