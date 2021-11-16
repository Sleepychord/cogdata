# -*- encoding: utf-8 -*-
'''
@File    :   txt_dataset.py
@Time    :   2021/11/15 21:16:05
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from torch.utils.data import Dataset, IterableDataset
from cogdata.utils.register import register


def _line_count(path):
    with open(path, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(0x4000000)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1 # Remove this if a wc-like count is needed
    return count

@register
class StreamingTxtDataset(IterableDataset):
    def __init__(self, path, world_size=1, rank=0, transform_fn=None):
        """
        Parameters
        ----------
        path:str
            The path of the txt file.
        world_size:int
            The total number of GPUs
        rank:int
            The local rank of current process
        transform_fn:function
            
        """
        self.path = path
        self.transform_fn = transform_fn
        # new handle
        self.handle = None
        self.pointer = 0
        self.world_size = self.all_world_size = world_size
        self.rank = self.all_rank = rank
        self.length = _line_count(path)
        self.length = self.length // world_size + \
            int(self.length % world_size > 0 and self.length % world_size <= rank + 1)

    def __len__(self):
        """Get the total number of the valid samples.

        Returns
        -------
        int
            The total number of the valid samples.
        """
        return self.length

    def __next__(self):
        """Returns the next sample in the dataset
        """
        if self.handle is None:
            self.handle = open(self.path, 'r')
            skipped = self.all_rank
        else:
            skipped = self.all_world_size - 1 
        for i in range(skipped):
            _dump = self.handle.__next__()
        line = self.handle.__next__()
        
        return line.strip()

    def __iter__(self):
        """StreamingRarDataset is iterable
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pass
        else:
            self.all_world_size = self.world_size * worker_info.num_workers
            self.all_rank = self.rank * worker_info.num_workers + worker_info.id
        return self


    def __del__(self):
        """Close the rar file"""
        if self.handle is not None:
            self.handle.close()