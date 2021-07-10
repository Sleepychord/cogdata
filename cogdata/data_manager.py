# -*- encoding: utf-8 -*-

import os
import sys
import json
import math
import random

from utils.data import format_file_size

class DataManager():
    def __init__(self, args) -> None:
        self.current_dir = '/workspace/zwd/test_dir'
    
    def list(self):
        '''List all datasets in current dir.

        dataset1(233 MB) rar json processed(10MB)
        dataset2(10 GB) zip json_ks unprocessed
        --------------- Summary ---------------
        current taskname: image_text_tokenization
        number(2) raw(10.23GB) processed(10MB)
        unprocessed: dataset2 
        '''
        datasets = os.listdir(self.current_dir)
        unprocessed_names = []
        size_sum = 0
        processed_size_sum = 0
        cnt = 0

        for dataset in datasets:
            info = None
            path = os.path.join(self.current_dir, dataset)

            if not os.path.isdir(path):
                continue

            try:
                with open(os.path.join(path, 'cogdata_info.json'), 'r') as info_file:
                    info = json.load(info_file)

                assert 'name' in info and type(info['name']) is str
                assert 'data_files' in info and type(info['data_files']) is list
                assert 'data_format' in info and type(info['data_format']) is str
                assert 'text_format' in info and type(info['text_format']) is str
                assert 'processed_files' not in info or ('processed_files' in info and type(info['processed_files']) is list)

                print(f"{info['name']}", end = ' ')

                size = 0
                for file in info['data_files']:
                    size += os.path.getsize(os.path.join(path, file))
                print(f"({format_file_size(size)})", end = ' ')

                print(f"{info['data_format']}", end = ' ')
                print(f"{info['text_format']}", end = ' ')

                processed_size = 0
                if 'processed_files' in info and len(info['processed_files']) > 0:
                    for file in info['processed_files']:
                        processed_size += os.path.getsize(os.path.join(path, file))
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

    def new_dataset(self, dataset_name):
        '''Create a dataset subfolder and a template (cogdata_info.json) in it.
        One should manually handle the data files.
        '''
        pass

    def new_task(self, args):
        '''create a cogdata_workspace subfolder and cogdata_config.json with configs in args.
        '''
        pass

    def process(self, args):
        '''process one or some (in args) unprocessed dataset (detected).
        '''
        pass

    def merge(self, args):
        '''merge all current processed datasets.
        '''
        pass
    def split(self, args):
        '''split the merged files into N parts.
        '''
        pass
    


if __name__ == '__main__':
    manager = DataManager(1)
    manager.list()