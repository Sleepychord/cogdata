# -*- encoding: utf-8 -*-
'''
@File    :   test_txt_tokenization_task.py
@Time    :   2021/11/16 13:56:00
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import json
import shutil
import random
import torch
from cogdata.datasets import StreamingTxtDataset, BinaryDataset
from cogdata.data_savers import BinarySaver
from cogdata.tasks import BilingualTextTokenizationTask
from cogdata.utils.ice_tokenizer import get_tokenizer

def test_image_text_tokenization_task():
    test_dir = 'tmp/test_txt_tokenization_task'
    case_dir = 'downloads/testcase/test_txt_tokenization_task'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    model_path = 'downloads/vqvae_hard_biggerset_011.pt'
    saver = BinarySaver(os.path.join(test_dir, 'testcase.bin'))
    task = BilingualTextTokenizationTask(saver=saver)
    ds = StreamingTxtDataset(os.path.join(case_dir, 'wiki1k.txt'), 
        transform_fn=task.get_transform_fn()
    )
    task.process([ds], 
        dataloader_num_workers=2,
        ratio=1,
        model_path=model_path
        )

    bin_ds = BinaryDataset(os.path.join(test_dir, 'testcase.bin'), length_per_sample=100, dtype='int32', preload=True, drop_last=True)
    tokenizer = get_tokenizer()
    
    for sample in bin_ds[:10]:
        print(tokenizer.decode(sample))
    first_line = tokenizer.decode(bin_ds[0])
    assert first_line.split()[1] == 'Cosmonaut'