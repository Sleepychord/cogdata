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