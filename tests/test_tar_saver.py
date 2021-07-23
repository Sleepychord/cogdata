# -*- encoding: utf-8 -*-
'''
@File    :   test_tar_saver.py
@Time    :   2021/07/13 21:55:20
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import shutil
import random

from cogdata.datasets import ZipDataset

from cogdata.data_savers import TarSaver

def test_tar_saver():
    test_dir = 'tmp/test_tar_saver'
    case_dir = 'downloads/testcase/test_tar_saver'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    output_path = test_dir
    saver = TarSaver(os.path.join(output_path, 'conv.tar'))
    def extract_fn(fp, full_filename, filesize):
        saver.save(fp, full_filename, filesize)
        return None
    ds = ZipDataset(os.path.join(case_dir, 'testcase.zip'), transform_fn=extract_fn)
    print([x for x in ds])
    del saver