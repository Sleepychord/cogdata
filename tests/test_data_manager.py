# -*- encoding: utf-8 -*-

import os
import sys
import math
import json
import time
import random
import torch
import shutil
import argparse

from cogdata.data_manager import DataManager

def test_data_manager():
    test_dir = 'tmp/test_data_manager'
    case_dir = '../../downloads/testcase/test_data_manager'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    os.chdir(test_dir)

    # test new
    args = argparse.Namespace()
    args.name = 'd0'
    args.description = 'd0'
    args.data_files = ['testcase.tar']
    args.data_format = 'TarDataset'
    args.text_files = ['testcase.json']
    args.text_format = 'json_ks'
    os.chdir('../test_data_manager')
    DataManager.new_dataset(args)
    shutil.copyfile(os.path.join(case_dir, 'testcase.tar'), 'd0/testcase.tar')
    shutil.copyfile(os.path.join(case_dir, 'testcase.json'), 'd0/testcase.json')

    args.name = 'd1'
    args.description = 'd1'
    os.chdir('../test_data_manager')
    DataManager.new_dataset(args)
    shutil.copyfile(os.path.join(case_dir, 'testcase.tar'), 'd1/testcase.tar')
    shutil.copyfile(os.path.join(case_dir, 'testcase.json'), 'd1/testcase.json')

    all_datasets = DataManager.fetch_datasets('.')
    assert set(all_datasets) == set(['d0', 'd1'])
    with open('d1/cogdata_info.json', 'r') as fin:
        info = json.load(fin)
    assert vars(args) == info    
    args.datasets = ''
    # # test list w/o task
    # args = argparse.Namespace()
    # args.task_id = None
    # DataManager.list(args)

    # test new task
    args = argparse.Namespace()
    args.task_id = 'task0'
    args.saver_type = 'BinarySaver'
    args.task_type = 'ImageTextTokenizationTask'
    args.description = 'testcase tokenization'
    
    args.txt_len = 64
    args.model_path = 'downloads/vqvae_hard_biggerset_011.pt'
    args.img_sizes = [256]

    args.length_per_sample = 1088
    args.dtype = 'int32'
    os.chdir('../test_data_manager')
    DataManager.new_task(args)

    with open('cogdata_task_task0/cogdata_config.json', 'r') as fin:
        info = json.load(fin)
    del args.task_id
    assert vars(args) == info    
    args.datasets = ''

    # # test list with task
    # args = argparse.Namespace()
    # args.task_id = 'task0'
    # DataManager.list(args)

    # test process
    args = argparse.Namespace()
    args.task_id = 'task0'
    args.datasets = None#['d0']
    args.nproc = 2
    DataManager.process(args)
    
    # args = argparse.Namespace()
    # args.task_id = 'task0'
    # DataManager.list(args)

    # test merge
    args = argparse.Namespace()
    args.datasets = ''
    args.task_id = 'task0'
    DataManager.merge(args)
    assert os.path.exists('cogdata_task_task0/merge.bin')

    # test split
    args = argparse.Namespace()
    args.datasets = ''
    args.task_id = 'task0'
    args.n = 3
    DataManager.split(args)

    os.chdir('../..')