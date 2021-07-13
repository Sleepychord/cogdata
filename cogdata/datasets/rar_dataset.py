# -*- encoding: utf-8 -*-
'''
@File    :   rar_dataset.py
@Time    :   2021/07/10 16:00:15
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

from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets
from PIL import Image
import timeit
import torch.distributed as dist
from ..utils.logger import get_logger


local_unrarlib_path = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    ), 'lib', 'libunrar.so'
)
if os.path.exists(local_unrarlib_path):
    os.environ['UNRAR_LIB_PATH'] = local_unrarlib_path
elif os.path.exists('/usr/local/lib/libunrar.so'):
    os.environ['UNRAR_LIB_PATH'] = '/usr/local/lib/libunrar.so'
elif os.path.exists('/usr/lib/libunrar.so'):
    os.environ['UNRAR_LIB_PATH'] = '/usr/lib/libunrar.so'

import unrar
from unrar import rarfile
from unrar import unrarlib
from unrar import constants
from unrar.rarfile import _ReadIntoMemory

class StreamingRarDataset(IterableDataset):
    def __init__(self, path, transform_fn=None):
        self.rar = rarfile.RarFile(path)
        self.transform_fn = transform_fn
        # new handle
        self.handle = None
        self.raw_members = [
            x for x in self.rar.namelist()
            if x[-1] != os.sep and '__MACOSX' not in x
        ]
        # split by distributed
        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            self.raw_members = [
                x for i, x in enumerate(self.raw_members) 
                if i % num_replicas == rank
            ]

    def __len__(self):
        return len(self.raw_members)

    def __next__(self):
        if self.pointer >= len(self.members):
            raise StopIteration()
        if self.handle == None:
            archive = unrarlib.RAROpenArchiveDataEx(
            self.rar.filename, mode=constants.RAR_OM_EXTRACT)
            self.handle = self.rar._open(archive)
        # callback to memory
        self.data_storage = _ReadIntoMemory()
        c_callback = unrarlib.UNRARCALLBACK(self.data_storage._callback)
        unrarlib.RARSetCallback(self.handle, c_callback, 0)    
        handle = self.handle
        try:
            rarinfo = self.rar._read_header(handle)
            while rarinfo is not None:
                if rarinfo.filename == self.members[self.pointer]:
                    self.rar._process_current(handle, constants.RAR_TEST)
                    break
                else:
                    self.rar._process_current(handle, constants.RAR_SKIP)
                rarinfo = self.rar._read_header(handle)

            if rarinfo is None:
                self.data_storage = None

        except unrarlib.UnrarException:
            ret = None
            full_filename = self.members[self.pointer]
            get_logger().warning(f'{full_filename} is a bad rarfile.')
        else:
            ret = self.data_storage.get_bytes()
        
        if self.transform_fn is not None:
            ret = self.transform_fn(ret, self.members[self.pointer])

        self.pointer += 1
        return ret

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = self.raw_members
        else:
            all_members = self.raw_members
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]
        self.pointer = 0
        return self

    def __del__(self):
        self.rar._close(self.handle)
