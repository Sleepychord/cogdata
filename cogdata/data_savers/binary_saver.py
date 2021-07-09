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

from .base_saver import BaseSaver

class BinarySaver(BaseSaver):
    def __init__(self, output_path):
        self.bin = open(output_path, 'wb')
    
    def save(self, data):
        '''
        Args:
            data: tensor.
        '''
        self.bin.write(data.cpu().numpy().tobytes()) # TODO buffer

    def commit(self):
        pass
    
    def __del__(self):
        self.commit()
        self.bin.close()