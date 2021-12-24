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
import time
import json
import random
import shutil
import argparse
from cogdata.data_processor import DataProcessor
from cogdata.utils.logger import get_logger
def test_monitor():
    test_dir = 'tmp/test_monitor'
    case_dir = 'downloads/testcase/test_monitor'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    current_dir, taskid = test_dir, '0'

    task_dir = os.path.join(current_dir, f"cogdata_task_{taskid}")
    if os.path.exists(case_dir): 
        dataset_names = ['colorhub', '88tph']
    else:
        dataset_names = []
        get_logger().warning('Big test not exist, skipping...')

    for name in dataset_names:
        shutil.copytree(os.path.join(case_dir, name), os.path.join(current_dir, name))
        
        # set meta_info
        data_dir = os.path.join(task_dir, name)
        os.makedirs(data_dir)
        meta_info = {
            'create_time': time.strftime(
                "%Y-%m-%d %H:%M:%S %Z", time.localtime()),
            'state': 0
        }
        with open(os.path.join(data_dir, 'meta_info.json'), 'w') as meta_info_file:
            json.dump(meta_info, meta_info_file, indent = 4)

    args = argparse.Namespace()
    args.task_id = taskid
    args.nproc = 4
    args.saver_type = 'BinarySaver'
    args.task_type = 'ImageTextTokenizationTask'
    args.batch_size = 32
    args.dataloader_num_workers = 8
    args.txt_len = 64
    args.img_sizes = [256]
    args.model_path = 'downloads/vqvae_hard_biggerset_011.pt'
    args.datasets = dataset_names
    # args.datasets = None
    
    dp = DataProcessor()
    dp.run_monitor(current_dir, taskid, args)

    # do assert
    for name in dataset_names:
        task_dir = os.path.join(current_dir,'cogdata_task_0')
        files = os.listdir(os.path.join(task_dir, name))
        for i in range(4):
            assert f"{name}.bin.part_{i}.cogdata" in files
