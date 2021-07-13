# -*- encoding: utf-8 -*-
'''
@File    :   test_zip_rar_tar_datasets.py
@Time    :   2021/07/13 19:12:51
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import PIL

from cogdata.datasets import ZipDataset, TarDataset, StreamingRarDataset

def extract_fn(fp, full_filename):
    name = os.path.split(full_filename)[-1]
    img = PIL.Image.open(fp).convert('RGB')
    img.save(os.path.join('tmp', current_ds, name))
    return name

def test_zip_dataset():
    global current_ds
    current_ds = 'zip'
    os.makedirs(os.path.join('tmp', current_ds), exist_ok=True)
    ds = ZipDataset('downloads/testcase.zip', extract_fn)
    print([x for x in ds])

def test_tar_dataset():
    global current_ds
    current_ds = 'tar'
    os.makedirs(os.path.join('tmp', current_ds), exist_ok=True)
    ds = TarDataset('downloads/testcase.tar', extract_fn)
    print([x for x in ds])
    
def test_rar_dataset():
    global current_ds
    current_ds = 'rar'
    os.makedirs(os.path.join('tmp', current_ds), exist_ok=True)
    ds = StreamingRarDataset('downloads/testcase.rar', extract_fn)
    print([x for x in ds])
    