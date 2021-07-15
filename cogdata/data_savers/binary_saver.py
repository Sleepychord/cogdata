# -*- encoding: utf-8 -*-
'''
@File    :   binary_saver.py
@Time    :   2021/07/09 19:57:12
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from tqdm import tqdm
from .base_saver import BaseSaver
from cogdata.utils.register import register

@register
class BinarySaver(BaseSaver):
    suffix = '.bin'
    max_buffer_size = 1024 * 1024 * 1024 * 10

    def __init__(self, output_path, dtype='int32', **kwargs):
        self.bin = open(output_path, 'wb', buffering=-1) # TODO test speed of buffering
        mapping = {'int32': torch.IntTensor,
                   'int64': torch.LongTensor,
                   'float32': torch.FloatTensor,
                   'uint8': torch.ByteTensor,
                   'bool': torch.BoolTensor
                   }
        self.dtype = mapping[dtype]

    def save(self, data):
        '''
        Args:
            data: tensor.
        '''
        self.bin.write(data.type(self.dtype).numpy().tobytes())  # TODO buffer

    def commit(self):
        self.bin.flush()
    
    def __del__(self):
        self.commit()
        self.bin.close()

    @classmethod
    def merge(cls, files, output_path, overwrite=False):
        '''
        Merge files into one.
        '''
        if os.path.exists(output_path):
            if overwrite:
                os.remove(output_path)
            else:
                raise FileExistsError
        ret = os.system('cat {} >> {}'.format(
            ' '.join(files), output_path
        )) # TODO: solve possible "Argument list too long"
        if ret != 0:
            raise Exception(f'cat return code {ret}')

    @classmethod
    def merge_file_read(cls, files, output_path, overwrite=False):
        merge_file = open(output_path, 'wb')

        for file_path in tqdm(files):
            with open(file_path, 'rb') as data_file:
                while True:
                    data = data_file.read(cls.max_buffer_size)
                    if not data:
                        break
                    merge_file.write(data)

        merge_file.close()
    
    @classmethod
    def split(cls, input_path, output_dir, n):
        '''
        Split input_path into n files in output_path.
        '''
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, os.path.split(input_path)[-1]+'.part')
        ret = os.system('split -d {} -n {} {} -a 3'.format(
            input_path, n, prefix
        ))
        if ret != 0:
            raise Exception(f'split return code {ret}')

    @classmethod
    def split_file_read(cls, input_path, output_dir, n, size):
        with open(input_path, 'rb') as merge_file:
            for i in tqdm(range(n - 1)):
                left_size = size
                merge_trunk = open(os.path.join(output_dir, os.path.split(input_path)[-1]+f".part{i}"), 'wb')
                while left_size > cls.max_buffer_size:
                    merge_trunk.write(merge_file.read(cls.max_buffer_size))
                    left_size -= cls.max_buffer_size
                merge_trunk.write(merge_file.read(left_size))
                merge_trunk.close()

            merge_trunk = open(os.path.join(output_dir, os.path.split(input_path)[-1]+f".part{n - 1}"), 'wb')
            while True:
                data = merge_file.read(cls.max_buffer_size)
                if not data:
                    break
                merge_trunk.write(data)          
            merge_trunk.close()
