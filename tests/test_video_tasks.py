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
from cogdata.tasks import VideoSceneTextTokenizationTask, IcetkVideoTextTokenizationTask
from cogdata.utils.cogview import get_tokenizer
from icetk import icetk as ice_tokenizer


def test_video_scene_text_tokenization_task():
    test_dir = 'tmp/test_video_scene_text_tokenization_task'
    case_dir = 'downloads/testcase/test_video_scene_text_tokenization_task'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    model_path = 'downloads/vqvae_hard_biggerset_011.pt'
    saver = BinarySaver(os.path.join(test_dir, 'testcase_quanjing.bin'))
    task = VideoSceneTextTokenizationTask(img_sizes=[128, 256], saver=saver, frame_num=7, max_clip_per_video=32)
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
        model_path=model_path
        )

    with open(os.path.join(case_dir, 'video_quanjing_data.json')) as testcase_text_file:
        testcase_text_dic = json.load(testcase_text_file)
    testcases = testcase_text_dic["RECORDS"]
    text0, text2 = testcases[0]["shortText"], testcases[2]["shortText"]

    bin_ds = BinaryDataset(os.path.join(test_dir, 'testcase_quanjing.bin'), length_per_sample=(32*32+16*16)*7+64, dtype='int32', preload=True)
    tokenizer = get_tokenizer()
    for sample_id in range(len(bin_ds)):
        x = 0
        while bin_ds[sample_id][x] != -1 and x < 64:
            x+=1
        print(f"[TEXT{sample_id}] ", tokenizer.DecodeIds(bin_ds[sample_id][:x])[0][0])
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
        imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64+256*i:64+256*(i+1)].to('cuda')) for i in range(7)], dim=0)
        save_image(imgs, os.path.join(test_dir, f'testcase_output_case{xi}_scale16.jpg'), normalize=True)
        imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(x[64+256*7+1024*i:64+256*7+1024*(i+1)].to('cuda')) for i in range(7)], dim=0)
        save_image(imgs, os.path.join(test_dir, f'testcase_output_case{xi}_scale32.jpg'), normalize=True)
        
def test_icetk_video_text_tokenization_task():
    test_dir = 'tmp/test_icetk_video_text_tokenization_task'
    case_dir = 'downloads/testcase/test_video_scene_text_tokenization_task'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    saver = BinarySaver(os.path.join(test_dir, 'testcase_quanjing.bin'))
    task = IcetkVideoTextTokenizationTask(img_sizes=[160], saver=saver, frame_num=24, max_clip_per_video=6)
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

    bin_ds = BinaryDataset(os.path.join(test_dir, 'testcase_quanjing.bin'), length_per_sample=(20*20)*24+64, dtype='int32', preload=True)
    for sample_id in range(len(bin_ds)):
        x = 0
        while bin_ds[sample_id][x] != ice_tokenizer['<pad>'] and x < 64:
            x+=1
        print(f"[TEXT{sample_id}] ", ice_tokenizer.decode(bin_ds[sample_id][:x]))
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
        imgs = torch.cat([ice_tokenizer.decode(image_ids=x[64+400*i:64+400*(i+1)].to('cuda'), compress_rate=8) for i in range(24)], dim=0)
        save_image(imgs, os.path.join(test_dir, f'testcase_output_case{xi}_scale20.jpg'), normalize=True)
    
    
if __name__ == "__main__":
    test_icetk_video_text_tokenization_task()