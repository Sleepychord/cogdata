# -*- encoding: utf-8 -*-
'''
@File    :   test_txt_datasets.py
@Time    :   2021/11/15 22:25:44
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from cogdata.datasets import StreamingTxtDataset

def txt_collate_fn(x):
    return x

def test_streamingtxtdataset():
    with open('tmp/test_file.txt', 'w') as fout:
        for i in range(1000):
            fout.write(str(i) * 10 + '\n')
    d = StreamingTxtDataset('tmp/test_file.txt', world_size=2, rank=1)
    for i, l in enumerate(d):    
        assert l == str(i*2+1) * 10
    del d
    d = StreamingTxtDataset('tmp/test_file.txt', world_size=1, rank=0)
    loader = torch.utils.data.DataLoader(d, batch_size=4, shuffle=False,
                                num_workers=2, collate_fn=txt_collate_fn, pin_memory=True)
    assert len(set(sum([x for x in loader], []))) == 1000
        