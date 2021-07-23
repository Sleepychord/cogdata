# -*- encoding: utf-8 -*-
'''
@File    :   test_binary_saver_dataset.py
@Time    :   2021/07/13 15:04:25
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import shutil
import random
import torch
from cogdata.datasets import BinaryDataset
from cogdata.data_savers import BinarySaver

def test_save_and_load():
    test_dir = 'tmp/test_save_and_load'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    output_path = os.path.join(test_dir, 'fake.bin')

    saver = BinarySaver(output_path, dtype='int32')
    d = torch.arange(100)
    saver.save(d)
    saver.commit()

    dataset = BinaryDataset(output_path, length_per_sample=5, dtype='int32', preload=True)
    assert dataset[3][2] == 17
    assert (dataset[-1] == d[-5:]).all()

    dataset = BinaryDataset(output_path, length_per_sample=5, dtype='int32', preload=False)
    assert dataset[3][2] == 17
    assert (dataset[-1] == d[-5:]).all()

def test_binary_merge_and_split():
    test_dir = 'tmp/test_binary_merge_and_split'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    output_path1 = os.path.join(test_dir, 'fake1.bin')
    saver = BinarySaver(output_path1, dtype='int32')
    d1 = torch.arange(500)
    saver.save(d1)
    saver.commit()

    output_path2 = os.path.join(test_dir, 'fake2.bin')
    saver = BinarySaver(output_path2, dtype='int32')
    d2 = torch.arange(500)
    saver.save(d2)
    saver.commit()

    output_path_merge = os.path.join(test_dir, 'merge.bin')
    BinarySaver.merge([output_path1, output_path2], output_path_merge, overwrite=True)
    dataset = BinaryDataset(output_path_merge, length_per_sample=5, dtype='int32', preload=True)
    assert dataset[3][2] == 17
    assert (dataset[-1] == d2[-5:]).all()

    split_path = os.path.join(test_dir, 'split_merge')
    os.makedirs(split_path, exist_ok=True)
    BinarySaver.split(output_path_merge, split_path, 5, dtype='int32', length_per_sample=5)
    dataset = BinaryDataset(os.path.join(split_path, 'merge.bin.part1'), length_per_sample=5, dtype='int32', preload=False)
    assert dataset[3][2] == 217
    assert (dataset[-1] == d1[395:400]).all()
