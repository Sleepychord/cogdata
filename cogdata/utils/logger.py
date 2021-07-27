# -*- encoding: utf-8 -*-
# @File    :   logger.py
# @Time    :   2021/07/10 21:16:05
# @Author  :   Ming Ding 
# @Contact :   dm18@mail.tsinghua.edu.cn

# here put the import lib
import os
import sys
import math
import random

import logging

__all__ = ['get_logger', 'set_logger']


def set_logger(target_path, rank='main'):
    """Set a specific logger for the current process

    Parameters
    ----------
    target_path:str
        The root folder of all log files
    rank:str
        The local rank of the current process
    """
    global logger
    if target_path is not None:
        logger = logging.getLogger(f'cogdata rank_{rank}')
        logger.setLevel(logging.DEBUG)
        # logger.propagate = False

        logfile = os.path.join(target_path, f'rank_{rank}.log')
        os.makedirs(os.path.dirname(logfile), exist_ok=True)

        fh = logging.FileHandler(logfile, mode='w')
        # fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


logger = logging.root
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)


def get_logger():
    """Returns a logger that is set by ``set_logger``"""
    return logger
