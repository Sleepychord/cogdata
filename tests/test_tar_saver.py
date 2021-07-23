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
import random

from cogdata.datasets import ZipDataset

from cogdata.data_savers import TarSaver

def test_tar_saver():
    output_path = os.path.join('tmp', 'tar_saver')
    os.makedirs(output_path, exist_ok=True)
    saver = TarSaver(os.path.join(output_path, 'conv.tar'))
    def extract_fn(fp, full_filename, filesize):
        saver.save(fp, full_filename, filesize)
        return None
    ds = ZipDataset('downloads/testcase.zip', transform_fn=extract_fn)
    print([x for x in ds])
    del saver