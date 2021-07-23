# -*- encoding: utf-8 -*-
'''
@File    :   test_image_text_tokenization_task.py
@Time    :   2021/07/14 20:51:57
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import json
import shutil
import random
import torch
from cogdata.datasets import TarDataset, BinaryDataset
from cogdata.data_savers import BinarySaver
from cogdata.tasks import ImageTextTokenizationTask
from cogdata.utils.cogview import get_tokenizer

def test_image_text_tokenization_task():
    test_dir = 'tmp/test_image_text_tokenization_task'
    case_dir = 'downloads/testcase/test_image_text_tokenization_task'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    model_path = 'downloads/vqvae_hard_biggerset_011.pt'
    saver = BinarySaver(os.path.join(test_dir, 'testcase.bin'))
    task = ImageTextTokenizationTask(img_sizes=[256, 512, 128], saver=saver)
    ds = TarDataset(os.path.join(case_dir, 'testcase.tar'), 
        transform_fn=task.get_transform_fn()
    )
    task.process([ds], 
        text_files=[os.path.join(case_dir, 'testcase.json')], 
        text_format='json_ks',
        device='cuda',
        dataloader_num_workers=2,
        txt_len=64,
        ratio=1,
        model_path=model_path
        )

    with open(os.path.join(case_dir, 'testcase.json')) as testcase_text_file:
        testcase_text_dic = json.load(testcase_text_file)
    testcases = testcase_text_dic["RECORDS"]
    text0, text2 = testcases[0]["cnShortText"], testcases[2]["cnShortText"]

    bin_ds = BinaryDataset(os.path.join(test_dir, 'testcase.bin'), length_per_sample=64*64+32*32+16*16+64, dtype='int32', preload=True)
    tokenizer = get_tokenizer()
    x = 0
    while bin_ds[0][x] != -1 and x < 64:
        x += 1
    assert text0 == tokenizer.DecodeIds(bin_ds[0][:x])[0][0]

    x = 0
    while bin_ds[2][x] != -1 and x < 64:
        x += 1
    assert text2 == tokenizer.DecodeIds(bin_ds[2][:x])[0][0]

    from torchvision.utils import save_image
    imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64:64+64**2].to('cuda')) for x in bin_ds], dim=0)
    save_image(imgs, os.path.join(test_dir, 'testcase512.jpg'), normalize=True)
    imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64+64**2:64+64**2+32**2].to('cuda')) for x in bin_ds], dim=0)
    save_image(imgs, os.path.join(test_dir, 'testcase256.jpg'), normalize=True)
    imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64+64**2+32**2:].to('cuda')) for x in bin_ds], dim=0)
    save_image(imgs, os.path.join(test_dir, 'testcase128.jpg'), normalize=True)