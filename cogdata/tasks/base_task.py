# -*- encoding: utf-8 -*-
'''
@File    :   base_task.py
@Time    :   2021/07/09 22:13:57
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self) -> None:
        '''config saver
        '''
        raise NotImplementedError

    @abstractmethod
    def get_transform_fn(self):
        def transform_fn(*args):
            '''Run in dataloader subprocess
            '''
            raise NotImplementedError
        return transform_fn
    
    @abstractmethod
    def process(self, *args):
        '''Use cuda to process batch data from dataloader,
            save via Saver,
            report progress every 1/5000 ?
            final commit saver
        '''
        raise NotImplementedError

    def display(self, *args):
        '''Display samples of this task.
        '''
        raise NotImplementedError