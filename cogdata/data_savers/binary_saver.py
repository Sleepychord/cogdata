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
from .base_saver import BaseSaver

class BinarySaver(BaseSaver):
    def __init__(self, output_path, dtype='int32'):
        self.bin = open(output_path, 'wb')
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
        self.bin.write(data.type(self.dtype).numpy().tobytes()) # TODO buffer

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