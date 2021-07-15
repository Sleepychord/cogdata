# -*- encoding: utf-8 -*-
'''
@File    :   progress_record.py
@Time    :   2021/07/09 22:06:19
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

class ProgressRecord():
    def __init__(self, log_dir, rank):
        self.rank = rank
        file_path = os.path.join(log_dir, f'rank_{rank}.progress')
        self.file = open(file_path, 'w')
    def __del__(self):
        self.file.close()
    def update(self, x, y):
        self.file.seek(0, 0) # beginning
        self.file.write(f'{x}/{y}')
        self.file.truncate()
    @classmethod
    def get_all(cls, log_dir, n):
        ret = []
        for rank in range(n):
            try:
                file_path = os.path.join(log_dir, f'rank_{rank}.progress')
                with open(file_path, 'r') as fin:
                    s = fin.read().split('/')
                    ret.append((int(s[0]), int(s[1])))
            except (FileNotFoundError, IndexError, ValueError):
                ret.append((None, None))
        return ret
        