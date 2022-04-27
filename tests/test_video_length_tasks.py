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
from cogdata.tasks import IcetkVideoTextLengthTokenizationTask
from cogdata.utils.cogview import get_tokenizer
from icetk import icetk as ice_tokenizer


def test_icetk_video_text_length_tokenization_task():
    test_dir = 'tmp/test_icetk_video_text_length_05sec_tokenization_task'
    case_dir = 'downloads/testcase/test_video_scene_text_tokenization_task'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    saver = BinarySaver(os.path.join(test_dir, 'testcase_quanjing.bin'))
    task = IcetkVideoTextLengthTokenizationTask(img_sizes=[160], saver=saver, 
                                                frame_num_per_clip=5, max_clip_per_video=4, clip_length=0.5)
    ds = TarDataset(os.path.join(case_dir, 'testcase_quanjing.tar'), 
        transform_fn=task.get_transform_fn()
    )
    task.process([ds], 
        text_files=[os.path.join(case_dir, 'video_quanjing_data.json')], 
        text_format='json_quanjing',
        device='cuda',
        dataloader_num_workers=2,
        txt_len=64,
        ratio=1,
        )

    with open(os.path.join(case_dir, 'video_quanjing_data.json')) as testcase_text_file:
        testcase_text_dic = json.load(testcase_text_file)
    testcases = testcase_text_dic["RECORDS"]
    text0, text2 = testcases[0]["shortText"], testcases[2]["shortText"]

    bin_ds = BinaryDataset(os.path.join(test_dir, 'testcase_quanjing.bin'), length_per_sample=(20*20)*5+64, dtype='int32', preload=True)
    for sample_id in range(len(bin_ds)):
        pad_idx = bin_ds[sample_id][1:].tolist().index(ice_tokenizer['<pad>'])
        n_idx = bin_ds[sample_id][:].tolist().index(ice_tokenizer['<n>'])
        if bin_ds[sample_id][0] == ice_tokenizer['<pad>']:
            text = ice_tokenizer.decode(bin_ds[sample_id][1:n_idx]) + " [NOT FULL] " + ice_tokenizer.decode(bin_ds[sample_id][n_idx+1:pad_idx])
        else:
            text = ice_tokenizer.decode(bin_ds[sample_id][:n_idx]) + " [FULL] " + ice_tokenizer.decode(bin_ds[sample_id][n_idx+1:pad_idx])

        print(f"[TEXT{sample_id}] ", text)
    # x = 0
    # while bin_ds[2][x] != -1 and x < 64:
    #     x += 1
    # assert text2 == tokenizer.DecodeIds(bin_ds[2][:x])[0][0]

    from torchvision.utils import save_image
    # for i in range(7):
    #     imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64+256*i:64+256*(i+1)].to('cuda')) for x in bin_ds], dim=0)
    #     save_image(imgs, os.path.join(test_dir, f'testcase_output_{i}_scale16.jpg'), normalize=True)
    #     imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64+256*7+1024*i:64+256*7+1024*(i+1)].to('cuda')) for x in bin_ds], dim=0)
    #     save_image(imgs, os.path.join(test_dir, f'testcase_output_{i}_scale32.jpg'), normalize=True)
    for xi, x in enumerate(bin_ds):
        imgs = torch.cat([ice_tokenizer.decode(image_ids=x[64+400*i:64+400*(i+1)].to('cuda'), compress_rate=8) for i in range(5)], dim=0)
        save_image(imgs, os.path.join(test_dir, f'testcase_output_case{xi}_scale20.jpg'), normalize=True)
    
    
if __name__ == "__main__":
    test_icetk_video_text_length_tokenization_task()