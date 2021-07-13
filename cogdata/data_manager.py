# -*- encoding: utf-8 -*-

import os
import sys
import json
import math
import random

from data_processor import DataProcessor
from utils.helpers import format_file_size
from data_savers import BinarySaver

data_size = {
    'int32': 4,
}

class DataManager():

    def __init__(self, base_dir) -> None:
        # base_dir = '/workspace/zwd/test_dir'
        self.base_dir = base_dir

        self.current_dir = None
        self.current_id = None

        self.task = None
        self.saver = None
        self.length_per_sample = None
        self.dtype = None

        self.merged = False
        self.merge_split = 0

        # self.processor = DataProcessor(args)

        # datasets = ["test_ds", "test_ds2"]
        # for dataset in datasets:
        #     folder_path = os.path.join(base_dir, dataset)
        #     if not os.path.exists(folder_path):
        #         os.mkdir(folder_path)
        #     self.new_dataset(dataset, {
        #         "name": "example",
        #         "description": "Describe the dataset",
        #         "data_files": ["example"],
        #         "data_format": "zip",
        #         "text_file": "example.json",
        #         "text_format": "json"
        #     })

        # self.new_task("test_task", 
        # {
        #     "task": "image_text_tokenization",
        #     "saver": "binary",
        #     "length_per_sample": 1089,
        #     "dtype": "int32"
        # })
    
    def fetch_datasets(self):
        datasets = []

        items = os.listdir(self.base_dir)
        for item in items:
            path = os.path.join(self.base_dir, item)
            if not os.path.isdir(path) or path == self.current_dir:
                continue
            
            if os.path.exists(os.path.join(path, 'cogdata_info.json')):
                datasets.append(item)
        
        return datasets
        
    def fetch_processed_datasets(self, datasets):
        processed_datasets = []
        
        for dataset in datasets:
            path = os.path.join(self.current_dir, dataset)
            processed_path = os.path.join(path, 'processed.bin')
            meta_info_path = os.path.join(path, 'meta_info.json')
            if os.path.exists(meta_info_path) and os.path.exists(processed_path):
                processed_datasets.append(dataset)
        
        return processed_datasets

    def list(self):
        '''List all datasets in current dir.

        dataset1(233 MB) rar json processed(10MB)
        dataset2(10 GB) zip json_ks unprocessed
        --------------- Summary ---------------
        current taskname: image_text_tokenization
        number(2) raw(10.23GB) processed(10MB)
        unprocessed: dataset2 
        '''
        datasets = self.fetch_datasets()
        processed_datasets = self.fetch_processed_datasets(datasets)

        unprocessed_names = []
        size_sum = 0
        processed_size_sum = 0
        cnt = 0

        for dataset in datasets:
            info = None
            path = os.path.join(self.base_dir, dataset)
            try:
                with open(os.path.join(path, 'cogdata_info.json'), 'r') as info_file:
                    info = json.load(info_file)

                assert 'name' in info and type(info['name']) is str
                assert 'data_files' in info and type(info['data_files']) is list
                assert 'data_format' in info and type(info['data_format']) is str
                assert 'text_format' in info and type(info['text_format']) is str

                print(info['name'], end = ' ')

                size = 0
                for file in info['data_files']:
                    try:
                        size += os.path.getsize(os.path.join(path, file))
                    except:
                        continue
                print(f"({format_file_size(size)})", end = ' ')

                print(info['data_format'], end = ' ')
                print(info['text_format'], end = ' ')

                processed_size = 0
                if dataset in processed_datasets:
                    path = os.path.join(self.current_dir, dataset)
                    processed_path = os.path.join(path, 'processed.bin')
                    processed_size = os.path.getsize(processed_path)
                    print(f"processed({format_file_size(processed_size)})")
                else:
                    unprocessed_names.append(info['name'])
                    print("unprocessed")

                size_sum += size
                processed_size_sum += processed_size
                cnt += 1

            except:
                print(f"Error: bad info in {dataset}.")
        
        print('--------------- Summary ---------------')
        
        config = None
        try:
            with open(os.path.join(self.current_dir, 'cogdata_config.json'), 'r') as config_file:
                config = json.load(config_file)

            assert 'task' in config and type(config['task']) is str

            print(f"current taskname: {config['task']}")

            print(f"number({cnt})", end = ' ')
            print(f"raw({format_file_size(size_sum)})", end = ' ')
            print(f"processed({format_file_size(processed_size_sum)})")

            if len(unprocessed_names) > 0:
                print(f"unprocessed: {' '.join(unprocessed_names)}")
            
        except:
            print("Error: bad config.")

    # def new_dataset(self, dataset_name, args):
    def new_dataset(self, 
                    dataset_name, description, 
                    data_files, data_format,
                    text_file, text_format):
        '''Create a dataset subfolder and a template (cogdata_info.json) in it.
        One should manually handle the data files.
        '''

        path = os.path.join(self.base_dir, dataset_name)
        info_path = os.path.join(path, 'cogdata_info.json')

        if not os.path.exists(path):
            os.mkdir(path)
        
        if os.path.exists(info_path):
            while True:
                sign = input(f"Warning: dataset {dataset_name} already existed. Rewrite?(y/n)")
                sign = sign.strip(' ').lower()
                if sign == 'y':
                    break
                elif sign == 'n':
                    return

        info_dic = {
            "name": dataset_name,
            "description": description,
            "data_files": data_files,
            "data_format": data_format,
            "text_file": text_file,
            "text_format": text_format
        }
        with open(info_path, 'w') as info:
            json.dump(info_dic, info)

    def load_task(self, id):
        path = os.path.join(self.base_dir, id)
        config_path = os.path.join(path, 'cogdata_config.json')

        if not (os.path.exists(path) and os.path.exists(config_path)):
            print(f"Error: task {id} not exist. Load failed.")
            return False
        
        try:
            self.current_dir = path
            self.current_id = id

            with open(config_path, 'r') as config_file:
                config = json.load(config_file)

            self.task = config['task']
            self.saver = config['saver']
            self.length_per_sample = config['length_per_sample']
            self.dtype = config['dtype']
        except:
            print(f"Error: Load task {id} failed with bad parameters.")
            return False

        return True

    def new_task(self, id, task, saver, length_per_sample, dtype):
        '''create a cogdata_workspace subfolder and cogdata_config.json with configs in args.
        '''
        path = os.path.join(self.base_dir, id)
        config_path = os.path.join(path, 'cogdata_config.json')

        if not os.path.exists(path):
            os.mkdir(path)
        
        if os.path.exists(config_path):
            if self.current_dir is None:
                try:
                    self.current_dir = path
                    self.current_id = id

                    with open(config_path, 'r') as config_file:
                        config = json.load(config_file)

                    self.task = config['task']
                    self.saver = config['saver']
                    self.length_per_sample = config['length_per_sample']
                    self.dtype = config['dtype']
                except:
                    pass
            print(f"Error: Workspace {name} already existed. Setup failed.")
            return

        self.task = task
        self.saver = saver
        self.length_per_sample = length_per_sample
        self.dtype = dtype

        config_dic = {
            "task": task,
            "saver": saver,
            "length_per_sample": length_per_sample,
            "dtype": dtype,
        }
        with open(config_path, 'w') as config:
            json.dump(config_dic, config)   

        self.current_dir = path  
        self.current_id = id  

    def process(self, args):
        '''process one or some (in args) unprocessed dataset (detected).
        '''
        datasets = ["example"]
        # run process with data_processor api

    def merge(self):
        '''merge all current processed datasets.
        '''
        if self.merged is True and self.merge_split > 0:
            for i in range(self.merge_split):
                os.remove(os.path.join(self.current_dir, f"merge_{i}.bin"))

        merge_file = open(os.path.join(self.current_dir, 'merge.bin'), 'wb')

        for dataset in self.processed_datasets:
            with open(os.path.join(self.current_dir, f"processed.{dataset}.bin"), 'rb') as data_file:
                merge_file.write(data_file.read())

        merge_file.close()
        self.merged = True

    def split(self, split_num):
        '''split the merged files into N parts.
        '''
        # split_num = 5

        if not self.merged:
            print(f"Error: task {self.task} not merged but receive a split request.")
            return
        
        sample_size = self.length_per_sample * data_size[self.dtype]

        if self.merge_split == 0:
            merge_path = os.path.join(self.current_dir, 'merge.bin')
            size = os.path.getsize(merge_path)
            sample_num = size // sample_size

            split_num = (sample_num + split_num - 1) // split_num

            with open(merge_path, 'rb') as merge_file:
                for i in range(split_num - 1):
                    merge_trunk = open(os.path.join(self.current_dir, f"merge_{i}.bin"), 'wb')
                    merge_trunk.write(merge_file.read(split_num * sample_size))
                    merge_trunk.close()
                merge_trunk = open(os.path.join(self.current_dir, f"merge_{split_num - 1}.bin"), 'wb')
                merge_trunk.write(merge_file.read())
                merge_trunk.close()

            os.remove(merge_path)

            print(f"Split task {self.task} successully.")
        else:
            print(f"Error: task {self.task} is already splitted.")
            return

if __name__ == '__main__':
    manager = DataManager(1)
    manager.list()
