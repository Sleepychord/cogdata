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
import random
import torch
from cogdata.datasets import TarDataset, BinaryDataset
from cogdata.data_savers import BinarySaver
from cogdata.tasks import ImageTextTokenizationTask
from cogdata.utils.cogview import get_tokenizer

def test_image_text_tokenization_task():
    model_path = '/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt'
    saver = BinarySaver('tmp/testcase.bin')
    task = ImageTextTokenizationTask(img_sizes=[256], saver=saver)
    ds = TarDataset('downloads/testcase.tar', 
        transform_fn=task.get_transform_fn()
    )
    task.process([ds], 
        text_files=['downloads/testcase.json'], 
        text_format='json_ks',
        device='cuda',
        dataloader_num_workers=2,
        txt_len=64,
        ratio=1,
        model_path=model_path
        )

    bin_ds = BinaryDataset('tmp/testcase.bin', length_per_sample=32*32+64, dtype='int32', preload=True)
    tokenizer = get_tokenizer()
    x = 0
    while bin_ds[0][x] != -1 and x < 64:
        x += 1
    assert "民国博物馆" == tokenizer.DecodeIds(bin_ds[0][:x])[0][0]

    x = 0
    while bin_ds[2][x] != -1 and x < 64:
        x += 1
    assert "佛山禅城祖庙古典风景摄影" == tokenizer.DecodeIds(bin_ds[2][:x])[0][0]

    from torchvision.utils import save_image
    print(bin_ds[0][64:])
    imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64:].to('cuda')) for x in bin_ds], dim=0)
    save_image(imgs, 'tmp/testcase.jpg', normalize=True)