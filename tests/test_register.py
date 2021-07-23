# -*- encoding: utf-8 -*-
'''
@File    :   test_register.py
@Time    :   2021/07/15 21:05:15
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from cogdata.utils.helpers import get_registered_cls

def test_register():
    test_dir = 'tmp/test_register'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    bin_saver_cls = get_registered_cls('BinarySaver')
    bin_ds_cls = get_registered_cls('BinaryDataset')
    
    output_path = os.path.join(test_dir, 'fake_register.bin')
    saver = bin_saver_cls(output_path, dtype='int32')
    d = torch.arange(100)
    saver.save(d)
    saver.commit()

    dataset = bin_ds_cls(output_path, length_per_sample=5, dtype='int32', preload=True)
    assert dataset[3][2] == 17
    assert (dataset[-1] == d[-5:]).all()
