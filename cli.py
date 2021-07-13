#!/usr/bin/env python

import argparse

from cogdata.data_manager import DataManager
from arguments import get_args

def add_task(manager, args):
    task_id = args.task_id
    task_type = args.task_type
    saver = args.saver
    length_per_sample = args.length_per_sample
    dtype = args.dtype

    try:
        assert not task_id is None
        assert not task_type is None
        assert saver in ['binary']
        assert length_per_sample > 0
        assert dtype in ['int32']
    except:
        print(f"Error: bad config for task {task_id}. Adding failed.")
        return

    manager.new_task(
        id = task_id,
        task = task_type,
        saver = saver,
        length_per_sample = length_per_sample,
        dtype = dtype
    )

def add_data(manager, args):
    name = args.dataset
    description = args.description
    data_files = args.data_files
    data_format = args.data_format
    text_file = args.text_file
    text_format = args.text_format

    try:
        assert not name in manager.fetch_datasets()
        assert data_format in ['zip', 'rar']
        assert text_format in ['json', 'json_ks']
    except:
        print(f"Error: bad infos for dataset {name}. Adding failed.")
        return

    manager.new_dataset(
        dataset_name = name,
        description = description,
        data_files = data_files,
        data_format = data_format,
        text_file = text_file,
        text_format = text_format
    )

def process(manager, args):
    task_id = args.task_id
    dataset = args.dataset

    if manager.load_task(task_id):
        if dataset is None:
            manager.process_all()
        else:
            manager.process(dataset)

def merge(manager, args):
    task_id = args.task_id

    if manager.load_task(task_id):
        manager.merge()

def split(manager, args):
    task_id = args.task_id
    split_num = args.split_num

    try:
        assert split_num > 1
    except:
        print(f"Error: incorrect split number. Failed")
        return
    
    if manager.load_task(task_id):
        manager.split(split_num)

def list_info(manager, args):
    task_id = args.task_id

    if manager.load_task(task_id):
        manager.list()

def display(manager, args):
    task_id = args.task_id
    dataset = args.dataset
    display_num = args.display_num

    if manager.load_task(task_id):
        manager.display(dataset, display_num)

if __name__ == '__main__':
    print('cli')
    # use argparse to finish cli
    args = get_args()
    base_dir = args.base_dir
    action = args.action

    manager = DataManager(base_dir)
    if action == 'add_task':
        add_task(manager, args)
    elif action == 'add_data':
        add_data(manager, args)
    elif action == 'process':
        process(manager, args)
    elif action == 'merge':
        merge(manager, args)
    elif action == 'split':
        split(manager, args)
    elif action == 'list':
        list_info(manager, args)
    elif action == 'display':
        display(manager, args)
    else:
        print("Please give a correct action.")
