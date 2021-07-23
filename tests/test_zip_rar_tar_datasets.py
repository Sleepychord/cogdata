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

def extract_fn(fp, full_filename, *args):
    name = os.path.split(full_filename)[-1]
    img = PIL.Image.open(fp).convert('RGB')
    img.save(os.path.join('tmp', current_ds, name))
    return name

def test_zip_dataset():
    global current_ds
    current_ds = 'test_zip_datasets'
    test_dir = os.path.join('tmp', current_ds)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    ds = ZipDataset('downloads/testcase/test_zip_rar_tar_datasets/testcase.zip', transform_fn=extract_fn)
    print([x for x in ds])

def test_tar_dataset():
    global current_ds
    current_ds = 'test_tar_datasets'
    test_dir = os.path.join('tmp', current_ds)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    ds = TarDataset('downloads/testcase/test_zip_rar_tar_datasets/testcase.tar', transform_fn=extract_fn)
    print([x for x in ds])
    
def test_rar_dataset():
    global current_ds
    current_ds = 'test_rar_datasets'
    test_dir = os.path.join('tmp', current_ds)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    ds = StreamingRarDataset('downloads/testcase/test_zip_rar_tar_datasets/testcase.rar', transform_fn=extract_fn)
    print([x for x in ds])
    