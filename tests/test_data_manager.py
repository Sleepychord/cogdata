# -*- encoding: utf-8 -*-

import os
import sys
import math
import json
import time
import random
import torch
import shutil

from cogdata.data_manager import DataManager

def test_task_init_load():
    test_base_dir = 'test_base_dir'
    if os.path.exists(test_base_dir):
        shutil.rmtree(test_base_dir)

    os.makedirs(test_base_dir, exist_ok=False)
    manager = DataManager("test_base_dir")

    id = str(1234)
    task = "image_text_tokenization"
    saver = "binary"
    length_per_sample = 1089
    image_length = 1024
    txt_length = 64
    dtype = "int32"

    manager.new_task(id, task, saver, length_per_sample, image_length, txt_length, dtype)
    assert manager.current_dir == os.path.join(test_base_dir, id)
    assert manager.current_id == id
    assert manager.task == task
    assert manager.saver == saver
    assert manager.length_per_sample == length_per_sample
    assert manager.image_length == image_length
    assert manager.txt_length == txt_length
    assert manager.dtype == dtype

    task_path = os.path.join(test_base_dir, id)
    config_path = os.path.join(task_path, 'cogdata_config.json')
    assert os.path.exists(config_path)
    with open (config_path, 'r') as config_file:
        config = json.load(config_file)
    
    assert task == config['task']
    assert saver == config['saver']
    assert length_per_sample == config['length_per_sample']
    assert image_length == config['image_length']
    assert txt_length == config['txt_length']
    assert dtype == config['dtype']

    manager.clear()

    manager.load_task(id)
    assert manager.current_dir == os.path.join(test_base_dir, id)
    assert manager.current_id == id
    assert manager.task == task
    assert manager.saver == saver
    assert manager.length_per_sample == length_per_sample
    assert manager.image_length == image_length
    assert manager.txt_length == txt_length
    assert manager.dtype == dtype

def test_init_data():
    test_base_dir = 'test_base_dir'
    os.makedirs(test_base_dir, exist_ok=False)
    manager = DataManager("test_base_dir")

    id = str(1234)
    task = "image_text_tokenization"
    saver = "binary"
    length_per_sample = 1089
    dtype = "int32"

    manager.new_task(id, task, saver, length_per_sample, dtype)

    name = "example"
    description = "Describe the dataset"
    data_files = ["example.zip"]
    data_format = "zip"
    text_file = "example.json"
    text_format = "json"

    manager.new_dataset(name, description, data_files, data_format, text_file, text_format)
    
    data_path = os.path.join(test_base_dir, name)
    info_path = os.path.join(data_path, 'cogdata_info.json')
    assert os.path.exists(info_path)
    with open(info_path, 'r') as info_file:
        info = json.load(info_file)
    
    assert name == info['name']
    assert description == info['description']
    assert data_files == info['data_files']
    assert data_format == info['data_format']
    assert test_file == info['text_file']
    assert text_format == info['text_format']

def test_fetch_data():
    test_base_dir = 'test_base_dir'
    os.makedirs(test_base_dir, exist_ok=False)
    manager = DataManager("test_base_dir")

    # construct a test datasets structure
    task_id = '1234'
    task_path = os.path.join(test_base_dir, task_id)
    os.makedirs(task_path)

    config = {
        "task": "image_text_tokenization",
        "saver": "binary",
        "length_per_sample": 1089,
        "dtype": "int32"
    }
    with open(os.path.join(task_path, 'cogdata_config.json'), 'w') as config_file:
        json.dump(config, config_file)
    
    for i in range(3):
        data_path = os.path.join(test_base_dir, f"dataset{i}")
        os.makedirs(data_path)
        with open(os.path.join(data_path, 'example.zip'), 'w') as example:
            example.write('test info')

        info = {
            "name": f"dataset{i}",
            "description": "Describe the dataset",
            "data_files": ["example.zip"],
            "data_format": "zip",
            "text_file": f"dataset{i}.json",
            "text_format": "json"
        }

        with open(os.path.join(data_path, 'cogdata_info.json'), 'w') as info_file:
            json.dump(info, info_file)
    
    for i in [1,2]:
        processed_data_path = os.path.join(task_path, f"dataset{i}")
        os.makedirs(processed_data_path)
        with open(os.path.join(processed_data_path, "meta_info.json"), 'w') as meta_info_file:
            meta_info = {
                'name': f"dataset{i}",
                'state': i - 1
            }
            json.dump(meta_info, meta_info_file)
        
        with open(os.path.join(processed_data_path, 'processed.bin'), 'w') as processed:
            processed.write('test data')

    datasets = manager.fetch_datasets()

    manager.load_task(task_id)
    processed_datasets = manager.fetch_processed_datasets(datasets)

    assert len(datasets) == 3 and len(processed_datasets) == 1
    for i in range(3):
        assert f"dataset{i}" in datasets
    assert processed_datasets[0] == 'dataset2'
    
def test_list():
    pass

def test_processed_merge_and_split():
    test_base_dir = 'test_base_dir'
    os.makedirs(test_base_dir, exist_ok=False)
    manager = DataManager("test_base_dir")

    # construct a test datasets structure
    task_id = '1234'
    task_path = os.path.join(test_base_dir, task_id)
    os.makedirs(task_path)

    config = {
        "task": "image_text_tokenization",
        "saver": "binary",
        "length_per_sample": 1,
        "image_length": 0,
        "txt_length": 0,
        "dtype": "int32"
    }
    with open(os.path.join(task_path, 'cogdata_config.json'), 'w') as config_file:
        json.dump(config, config_file)
    
    for i in range(3):
        data_path = os.path.join(test_base_dir, f"dataset{i}")
        os.makedirs(data_path)
        with open(os.path.join(data_path, 'example.zip'), 'w') as example:
            example.write('test info')

        info = {
            "name": f"dataset{i}",
            "description": "Describe the dataset",
            "data_files": ["example.zip"],
            "data_format": "zip",
            "text_file": f"dataset{i}.json",
            "text_format": "json"
        }

        with open(os.path.join(data_path, 'cogdata_info.json'), 'w') as info_file:
            json.dump(info, info_file)
    
    input_data = [
        b'testdata_example_piece00',
        b'testdata_example_piece01',
        b'testdata_example_piece02'
    ]
    for i in [0, 1, 2]:
        processed_data_path = os.path.join(task_path, f"dataset{i}")
        os.makedirs(processed_data_path)
        with open(os.path.join(processed_data_path, "meta_info.json"), 'w') as meta_info_file:
            meta_info = {
                'name': f"dataset{i}",
                'state': 1
            }
            json.dump(meta_info, meta_info_file)
        
        with open(os.path.join(processed_data_path, 'processed.bin'), 'wb') as processed:
            processed.write(input_data[i])
    
    manager.load_task(task_id)
    manager.merge()
    assert manager.merged == True

    with open(os.path.join(task_path, 'merge.bin'), 'rb') as data_file:
        data = data_file.read()
    assert data == b'testdata_example_piece00testdata_example_piece01testdata_example_piece02'
    
    manager.split(4)
    assert manager.merged == True
    assert manager.merge_split == 4

    expected_data = [
        b'testdata_example_pie',
        b'ce00testdata_example',
        b'_piece01testdata_exa',
        b'mple_piece02'
    ]
    for i in range(4):
        with open(os.path.join(task_path, f"merge_{i}.bin"), 'rb') as data_file:
            data = data_file.read()
        # print(data)
        assert data == expected_data[i]

def compare_merge_split():
    test_base_dir = '/dataset/fd5061f6/test_base_dir'
    manager = DataManager(test_base_dir)

    # construct a test datasets structure
    task_id = '1234'
    task_path = os.path.join(test_base_dir, task_id)

    # with open(os.path.join(task_path, 'cogdata_config.json'), 'r') as config_file:
    #     config = json.load(config_file)
    
    # config['merged'] = False
    # config['merge_split'] = 0
    # config['length_per_sample'] = 1089

    # with open(os.path.join(task_path, 'cogdata_config.json'), 'w') as config_file:
    #     json.dump(config, config_file, indent = 4)

    # manager.load_task(task_id)

    # print('merging with shell...')
    # start = time.time()
    # manager.merge_shell()
    # end = time.time()
    # print('merge time:', end - start)

    # assert manager.merged == True
    
    # print('splitting with shell...')
    # start = time.time()
    # manager.split_shell(10)
    # end = time.time()
    # print('split time:', end - start)

    # assert manager.merged == True
    # assert manager.merge_split == 10

    with open(os.path.join(task_path, 'cogdata_config.json'), 'r') as config_file:
        config = json.load(config_file)
    
    config['merged'] = False
    config['merge_split'] = 0
    config['length_per_sample'] = 1089

    with open(os.path.join(task_path, 'cogdata_config.json'), 'w') as config_file:
        json.dump(config, config_file, indent = 4)

    manager.load_task(task_id)

    print('merging...')
    start = time.time()
    manager.merge()
    end = time.time()
    print('merge time:', end - start)

    assert manager.merged == True
    
    print('splitting...')
    start = time.time()
    manager.split(10)
    end = time.time()
    print('split time:', end - start)

    assert manager.merged == True
    assert manager.merge_split == 10
