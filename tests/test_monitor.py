# -*- encoding: utf-8 -*-
'''
@File    :   test_monitor.py
@Time    :   2021/07/16 00:05:30
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import argparse
from cogdata.data_processor import DataProcessor
def test_monitor():
    dp = DataProcessor()
    current_dir, dataset_names, taskid = 'bigtest', ['enterdesk', 'nipic'], '0'
    args = argparse.Namespace()
    args.nproc = 4
    args.saver_type = 'BinarySaver'
    args.task_type = 'ImageTextTokenizationTask'
    args.batch_size = 32
    args.dataloader_num_workers = 8
    args.img_sizes = [256]
    args.model_path = '/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt'
    args.datasets = dataset_names
    dp.run_monitor(current_dir, taskid, args)