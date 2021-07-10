# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2021/07/10 21:16:05
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import logging

__all__ = ['get_logger', 'set_logger']

def set_logger(target_path):
    global logger
    import torch.distributed as dist
    if dist.is_available():
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)  # Log等级总开关

        rank = dist.get_rank()
        logfile = os.path.join(target_path, 'logs', f'rank{rank}.log')
        os.path.mkdir(os.path.dirname(logfile), exist_ok=True)

        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Log等级总开关

def get_logger():
    global logger
    return logger

