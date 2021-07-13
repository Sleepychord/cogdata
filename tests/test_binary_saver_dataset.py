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
import random
import torch
from cogdata.datasets import BinaryDataset
from cogdata.data_savers import BinarySaver

def test_save_and_load():
    os.makedirs('tmp', exist_ok=True)
    output_path = 'tmp/fake.bin'
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
