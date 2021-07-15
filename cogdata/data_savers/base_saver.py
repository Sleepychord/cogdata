# -*- encoding: utf-8 -*-
'''
@File    :   base_saver.py
@Time    :   2021/07/09 18:20:51
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

from abc import ABC, abstractmethod
class BaseSaver(ABC):
    suffix = '.dat'
    @abstractmethod
    def __init__(self, output_path, *args, **kwargs):
        pass
    @abstractmethod
    def save(self, *args):
        '''
        Save a sample, can be buffered.
        '''
        raise NotImplementedError
    @abstractmethod
    def commit(self):
        '''
        Commit all buffered samples.
        '''
        raise NotImplementedError
    @abstractmethod
    def __del__(self):
        '''
        Close and Commit.
        '''
        raise NotImplementedError

    @classmethod
    def merge(cls, files, output_path):
        '''
        Merge files into one.
        '''
        raise NotImplementedError
    
    @classmethod
    def split(cls, input_path, output_path, n):
        '''
        Split input_path into n files in output_path.
        '''
        raise NotImplementedError