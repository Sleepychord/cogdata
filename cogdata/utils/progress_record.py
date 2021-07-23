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
import time

class ProgressRecord():
    def __init__(self, log_dir, rank):
        self.rank = rank
        file_path = os.path.join(log_dir, f'rank_{rank}.progress')
        self.file = open(file_path, 'w')
        self.last_time = time.time()
        self.last_x = 0
    def __del__(self):
        self.file.close()
    def update(self, x, y, speed=None):
        if speed is None: # auto-detect
            current_time = time.time()
            speed =(x - self.last_x) / (current_time - self.last_time)
            self.last_time = current_time
            self.last_x = x
        self.file.seek(0, 0) # beginning
        self.file.write(f'{x}/{y}/{speed}')
        self.file.truncate()
    @classmethod
    def get_all(cls, log_dir, n):
        ret = []
        for rank in range(n):
            try:
                file_path = os.path.join(log_dir, f'rank_{rank}.progress')
                with open(file_path, 'r') as fin:
                    s = fin.read().split('/')
                    ret.append((int(s[0]), int(s[1]), float(s[2])))
            except (FileNotFoundError, IndexError, ValueError):
                ret.append((None, None, None))
        return ret
        