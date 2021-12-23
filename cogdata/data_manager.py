# -*- encoding: utf-8 -*-
"""This file defines DataManager.
"""

import os
import sys
import json
import math
import random
import shutil
import time
from tqdm import tqdm

from .data_processor import DataProcessor
from .utils.helpers import format_file_size, dir_size, get_registered_cls
from .utils.logger import get_logger


class DataManager():
    """A manager of all datasets.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def fetch_datasets(base_dir):
        """ Get all names of created datasets(dataset with ```config.json```) 

        Parameters
        ----------
        base_dir : str
            The root folder path 
        Returns
        -------
        list[str]
            A list of created dataset names.
        """
        datasets = []
        for item in os.listdir(base_dir):
            path = os.path.join(base_dir, item)
            if not os.path.isdir(path):
                continue
            if os.path.exists(os.path.join(path, 'cogdata_info.json')):
                datasets.append(item)
        return datasets

    @staticmethod
    def fetch_datasets_states(base_dir, task_id):
        """Get datasets status(dataset with ```config.json```) in the task which id is ```task_id```

        Args:
            base_dir(str): The root folder path 
            task_id(str): An ID of an exist task.


        Returns:
            tuple: a tuple containing:
                - all_datasets([str]): A list of created dataset names.
                - processed([str]): A list of processed dataset names.
                - hanging([str]): A list of processing dataset names.
                - unprocessed:([str]): A list of unprocessed dataset names.
                - additional:([str]) A list of only processed dataset names, from migration.
        """
        all_datasets = DataManager.fetch_datasets(base_dir)
        if task_id is None:
            return all_datasets, None, None, None, None
        processed, hanging, unprocessed = [], [], []
        task_path = os.path.join(base_dir, f'cogdata_task_{task_id}')
        for dataset in all_datasets:
            fld_path = os.path.join(task_path, dataset)
            meta_info_path = os.path.join(fld_path, 'meta_info.json')
            if not (os.path.exists(fld_path) and os.path.exists(meta_info_path)):
                unprocessed.append(dataset)
                continue
            with open(meta_info_path, 'r') as meta_info_file:
                meta_info = json.load(meta_info_file)
                state = meta_info.get('state', 0)
                if state == 1:
                    processed.append(dataset)
                else:
                    hanging.append(dataset)
        additional = []  # only processed results, from migration
        for item in os.listdir(task_path):
            path = os.path.join(task_path, item)
            if not os.path.isdir(path):
                continue
            meta_info_path = os.path.join(path, 'meta_info.json')
            if os.path.exists(meta_info_path):
                with open(meta_info_path, 'r') as meta_info_file:
                    meta_info = json.load(meta_info_file)
                    state = meta_info.get('state', 0)
                    if state == 1 and item not in processed:
                        additional.append(item)
        return all_datasets, processed, hanging, unprocessed, additional

    @staticmethod
    def list(args):
        """List all datasets in current dir

        Parameters
        ----------
        args:argparse.Namespace
            Arguments provided by the console

        Note
        ----
        dataset1(233 MB) rar json processed(10MB)
        dataset2(10 GB) zip json_ks unprocessed

        current taskname: image_text_tokenization
        number(2) raw(10.23GB) processed(10MB)
        unprocessed: dataset2 
        """
        base_dir = os.getcwd()
        task_id = args.task_id
        all_datasets, processed, hanging, unprocessed, additional = DataManager.fetch_datasets_states(
            base_dir, task_id)

        name_size_pair, sum_size = [], 0
        for dataset in all_datasets:
            path = os.path.join(base_dir, dataset)
            with open(os.path.join(path, 'cogdata_info.json'), 'r') as finfo:
                this_info = json.load(finfo)
                size = 0
                for file_name in this_info['data_files']:
                    data_path = os.path.join(path, file_name)
                    size += os.path.getsize(data_path)
            # size = dir_size(path)
            name_size_pair.append((dataset, format_file_size(size)))
            sum_size += size
        nsdict = dict(name_size_pair)
        print(
            '\n--------------------------- All Raw Datasets --------------------------    ')
        print(' '.join([f'{x}({y})' for x, y in name_size_pair]))

        print('------------------------------- Summary -------------------------------')
        print(
            f'Total {len(name_size_pair)} datasets\nTotal size: {format_file_size(sum_size)}')

        if task_id is None:
            return
        print('------------------------------ Task Info ------------------------------')

        task_path = os.path.join(base_dir, f'cogdata_task_{task_id}')
        try:
            with open(os.path.join(task_path, 'cogdata_config.json'), 'r') as config_file:
                config = json.load(config_file)
            assert 'task_type' in config and type(config['task_type']) is str
            print(f"Task Id: {task_id}")
            print(f"Task Type: {config['task_type']}")
            print(f'Description: {config["description"]}')
            print(
                f'\033[1;32mProcessed\033[0m:  FORMAT: dataset_name(raw_size => processed_size)')
            for dataset in processed:
                path = os.path.join(task_path, dataset)
                size = format_file_size(dir_size(path))
                print(f'{dataset}({nsdict[dataset]} => {size})', end=' ')
            print(
                '\n\033[1;33mHanging\033[0m:  FORMAT: dataset_name(raw_size)[create_time]')
            for dataset in hanging:
                meta_info_path = os.path.join(
                    task_path, dataset, 'meta_info.json')
                with open(meta_info_path, 'r') as fin:
                    info = json.load(fin)
                print(
                    f'{dataset}({nsdict[dataset]})[{info["create_time"]}]', end=' ')
            print('\nAdditional:  FORMAT: dataset_name(processed_size)')
            for dataset in additional:
                path = os.path.join(task_path, dataset)
                size = format_file_size(dir_size(path))
                print(f'{dataset}({size})', end=' ')
            print(
                '\n\033[1;31mUnprocessed\033[0m:  FORMAT: dataset_name(raw_size)')
            for dataset in unprocessed:
                print(f'{dataset}({nsdict[dataset]})', end=' ')
            print('')
        except Exception as e:
            print(e)
            print("Error: bad config.")

    @staticmethod
    def new_dataset(args):
        """Create a dataset subfolder and a template (cogdata_info.json) in it.
        One should manually handle the data files.

        Parameters
        ----------
        args:argparse.Namespace
            Arguments provided by the console
        """
        base_dir = os.getcwd()
        path = os.path.join(base_dir, args.name)
        info_path = os.path.join(path, 'cogdata_info.json')

        if not os.path.exists(path):
            os.mkdir(path)

        if os.path.exists(info_path):
            while True:
                sign = input(
                    f"Warning: dataset {args.name} already existed. Rewrite?(y/n)")
                sign = sign.strip(' ').lower()
                if sign == 'y':
                    break
                elif sign == 'n':
                    return
        # vars(args) = {
        #     "name": dataset_name,
        #     "description": description,
        #     "data_files": data_files,
        #     "data_format": data_format,
        #     "text_files": text_files,
        #     "text_format": text_format
        # }
        with open(info_path, 'w') as info:
            info_dict = vars(args).copy()
            info_dict.pop('func', None)
            json.dump(info_dict, info, indent=4)

    @staticmethod
    def load_task(base_dir, id):
        """Load task config(json) by task id

        Parameters
        ---------
        base_dir:str
            The root folder path 
        id:str 
            An ID of an exist task.

        Returns
        -------
        dict
            Config json of the task
        """
        path = os.path.join(base_dir, f"cogdata_task_{id}")
        config_path = os.path.join(path, 'cogdata_config.json')

        if not (os.path.exists(path) and os.path.exists(config_path)):
            raise ValueError(f"Error: task {id} not exist. Load failed.")

        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        if not ('task_type' in config and type(config['task_type']) is str):
            raise ValueError(
                f"Error: task {id} has no task_type config. Load failed.")
        return config

    @staticmethod
    def new_task(args):
        """create a cogdata_workspace subfolder and cogdata_config.json with configs in args.

        Parameters
        ----------
        args:argparse.Namespace
            Arguments provided by the console
        """
        base_dir = os.getcwd()
        id, task_type, saver_type = args.task_id, args.task_type, args.saver_type
        path = os.path.join(base_dir, f"cogdata_task_{id}")
        config_path = os.path.join(path, 'cogdata_config.json')

        if not os.path.exists(path):
            os.mkdir(path)

        if os.path.exists(config_path):
            print(f"Error: Workspace {id} already existed. Setup failed.")
            return

        config = vars(args).copy()
        config.pop('task_id', None)
        config.pop('func', None)
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

    @staticmethod
    def process(args):
        """process one or some (in args) unprocessed dataset (detected).

        Parameters
        ----------
        args:argparse.Namespace
            Arguments provided by the console
        """
        base_dir = os.getcwd()
        task_id = args.task_id
        all_datasets, processed, hanging, unprocessed, additional = DataManager.fetch_datasets_states(
            base_dir, task_id)

        if args.datasets is None or len(args.datasets) == 0:
            print('Processing all unprocessed datasets by default...')
            args.datasets = unprocessed
            if len(args.datasets) == 0:
                get_logger().warning('All datasets have been processed!')
                return

        # check valid
        loaded_task_args = DataManager.load_task(base_dir, task_id)
        args.__dict__.update(loaded_task_args)
        task_path = os.path.join(base_dir, f'cogdata_task_{args.task_id}')
        for name in args.datasets:
            assert name in unprocessed
            path = os.path.join(task_path, name)
            os.makedirs(path, exist_ok=True)
            meta_info = {
                'create_time': time.strftime(
                    "%Y-%m-%d %H:%M:%S %Z", time.localtime()),
                'state': 0
            }
            with open(os.path.join(path, 'meta_info.json'), 'w') as meta_info_file:
                json.dump(meta_info, meta_info_file, indent=4)
            if os.path.exists(os.path.join(path, 'logs')):
                shutil.rmtree(os.path.join(path, 'logs'))
        if hasattr(args, 'func'):  # no copy, may change later
            del args.func
        DataProcessor().run_monitor(base_dir, task_id, args)

    @staticmethod
    def process_single(args):
        """
        Parameters
        ----------
        args:argparse.Namespace
            Arguments provided by the console
        """
        DataProcessor().run_single(args.local_rank, json.loads(args.args_dict))

    @staticmethod
    def merge(args):  # TODO add additional target, not rm split
        """merge all current processed datasets.

        Parameters
        ----------
        args:argparse.Namespace
            Arguments provided by the console
        """
        base_dir = os.getcwd()
        task_id = args.task_id
        saver_type = DataManager.load_task(base_dir, task_id)['saver_type']
        task_path = os.path.join(base_dir, f'cogdata_task_{task_id}')
        split_path = os.path.join(task_path, 'split_merged_files')
        if os.path.exists(split_path):
            print('Removing previous split_merged_files...')
            shutil.rmtree(split_path)

        all_datasets, processed, hanging, unprocessed, additional = DataManager.fetch_datasets_states(
            base_dir, task_id)
        

        data_paths = []
        if args.datasets is None:
            datasets_to_merge = processed
            mid_name = ''
        else:
            datasets_to_merge = args.datasets
            mid_name = '_' + ','.join(datasets_to_merge)
            if len(mid_name) > 30:
                mid_name = mid_name[:30] + f'...{len(datasets_to_merge)}ds'
        print(f'Merging {datasets_to_merge} ...')
        
        merge_path = os.path.join(
            task_path, 'merge'+mid_name+get_registered_cls(saver_type).suffix)
        
        for dataset in datasets_to_merge:
            for item in os.listdir(os.path.join(task_path, dataset)):
                if item.endswith('.cogdata'):
                    data_paths.append(os.path.join(task_path, dataset, item))

        get_registered_cls(saver_type).merge(data_paths, merge_path, True)

    @staticmethod
    def split(args):
        """split the merged files into N parts.

        Parameters
        ----------
        args:argparse.Namespace
            Argument provided by the console
        """
        base_dir = os.getcwd()
        task_id = args.task_id
        task_config = DataManager.load_task(base_dir, task_id)
        task_config.update(args.__dict__)
        task_config.pop('n', None)
        task_config.pop('func', None)
        saver_type = task_config['saver_type']
        task_path = os.path.join(base_dir, f'cogdata_task_{task_id}')
        split_path = os.path.join(task_path, 'split_merged_files')
        merge_path = os.path.join(
            task_path, 'merge'+get_registered_cls(saver_type).suffix)
        if not os.path.exists(merge_path):
            print('Merged file not found. Failed.')
            return
        if os.path.exists(split_path):
            print(
                'Already split, mv or rm "split_merged_files" and try again for different N. Quitting...')
            return
        os.makedirs(split_path, exist_ok=True)

        get_registered_cls(saver_type).split(
            merge_path, split_path, args.n, **task_config)

    @staticmethod
    def clean(args):
        """Clean all files in a task subfolder

        Parameters
        ----------
        args:argparse.Namespace
            Argument provided by the console
        """
        base_dir = os.getcwd()
        task_id = args.task_id
        all_datasets, processed, hanging, unprocessed, additional = DataManager.fetch_datasets_states(
            base_dir, task_id)
        task_path = os.path.join(base_dir, f'cogdata_task_{task_id}')
        for dataset in hanging:
            shutil.rmtree(os.path.join(task_path, dataset))
        if len(hanging) > 0:
            get_logger().info(
                f'Remove (damaged) results of {" ".join(hanging)}!')
