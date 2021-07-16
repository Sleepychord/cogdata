# -*- encoding: utf-8 -*-
import os
import json
import time
import subprocess
import torch
from torchvision import transforms
from cogdata.utils.cogview.unified_tokenizer import get_tokenizer
from cogdata.utils.logger import get_logger, set_logger
from cogdata.utils.progress_record import ProgressRecord
from cogdata.eprogress import MultiProgressManager, LineProgress
from cogdata.utils.helpers import get_registered_cls
import logging
def initialize_distributed(local_rank, world_size, rank=None, 
        master_addr=None, master_port=None):
    """Initialize torch.distributed."""
    if rank is None:
        rank = local_rank
    device = local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    if master_addr is None:
        master_addr = os.getenv('MASTER_ADDR', 'localhost')
    if master_port is None:
        master_port = os.getenv('MASTER_PORT', '6005')
    init_method += master_addr + ':' + master_port
    get_logger().debug(f'rank {rank} setup distributed at {init_method}')
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size, rank=local_rank,
        init_method=init_method)


class DataProcessor():

    def __init__(self) -> None:
        pass

    def run_monitor(self, current_dir, taskid, args):
        '''Launch k run_single processes (by cmd, not multiprocess for dataloader)
           Monitor all the progresses by outputs in tmp files, clean tmp files from previous runs at first. use utils.progress_record !
           Wait and merge k files (use the helper in saver).
        '''
        dataset_names = args.datasets
        nproc = args.nproc
        task_path = os.path.join(current_dir, f'cogdata_task_{taskid}')
        script_path = 'manual_test_processor.py' # FIXME
        # already build meta_info, rm log dir from data_manager
        for name in dataset_names:
            log_dir = os.path.join(task_path, name, 'logs')
            os.makedirs(log_dir, exist_ok=False)
        del log_dir
        
        args_dict = vars(args)
        args_dict['_current_dir'] = current_dir
        args_dict['_task_path'] = task_path

        command = ["python", "-m", "torch.distributed.launch", 
            "--master_port", "30513", 
            "--nproc_per_node", str(nproc), 
            script_path, 
            "--args_dict", json.dumps(args_dict)
        ]
        print(command)
        progress_manager = MultiProgressManager()

        main_log_file = open(os.path.join(task_path, 'main_pid_{}.log'.format(os.getpid())), 'w')
        get_logger().info("All datasets: {}".format(' '.join(dataset_names)))
        progress_manager.skip_upline()
        world_size = nproc # TODO: multi-node
        for i in range(world_size):
            progress_manager.put(f'rank_{i}', 
                LineProgress(total=100, title=f'dataset: {dataset_names[0]}, rank {i}')
            ) 

        p = subprocess.Popen(command, stdout=main_log_file, stderr=main_log_file)
        try:
            last_progress = [(0, 1, 0)] * world_size
            ds_idx = 0
            get_logger().info(f'Processing {dataset_names[ds_idx]}')
            progress_manager.skip_upline()
            # logs for current dataset
            log_dir = os.path.join(task_path, dataset_names[ds_idx], 'logs') 
            finished = True # last dataset finished, examine the next
            while p.poll() is None or finished:
                time.sleep(0.5)
                if ds_idx >= len(dataset_names):
                    finished = False
                    continue
                # query progress
                collected_progress = ProgressRecord.get_all(log_dir, world_size)
                for i, (x, y, z) in enumerate(collected_progress):
                    if x is not None:
                        last_progress[i] = (x, y, z)
                # update bar
                for i in range(world_size):
                    x, y, z = last_progress[i]
                    progress_manager.update(f'rank_{i}', x * 100. / y, speed=z)
                # whether to next dataset
                finished = all([x == y > 0 for x,y,z in last_progress])
                if finished:
                    # set the state to 1
                    meta_info_path = os.path.join(task_path, dataset_names[ds_idx], 'meta_info.json')
                    with open(meta_info_path, 'r') as fin:
                        meta_info = json.load(fin)
                        meta_info['state'] = 1
                        meta_info['finish_time'] = time.strftime(
                            "%Y-%m-%d %H:%M:%S %Z", time.localtime())
                    with open(meta_info_path, 'w') as fout:
                        json.dump(meta_info, fout)
                         
                    ds_idx += 1
                    if ds_idx >= len(dataset_names):
                        s = f'Waiting torch.launch to terminate...'
                    else:
                        s = f'Processing {dataset_names[ds_idx]}'
                        log_dir = os.path.join(task_path, dataset_names[ds_idx], 'logs') 
                        for j in range(world_size):
                            progress_manager.update_title(f'rank_{j}', f'dataset: {dataset_names[ds_idx]}, rank {j}')
                    get_logger().info(s)
                    progress_manager.skip_upline()
                    last_progress = [(0, 1, 0)] * world_size
        except Exception as e:
            print(e)
            p.terminate()
        finally:
            main_log_file.close()
        if p.returncode > 0:
            get_logger().error(f'torch.launch returns code {p.returncode}')
        
        

    def run_single(self, local_rank, args_dict):
        '''really process, create datasets with task.transform_fn, iterating the dataloader and run task.process
        '''
        args_dict = args_dict.copy()
        rank = local_rank
        world_size = args_dict['nproc'] # TODO: multi-node
        current_dir = args_dict.pop('_current_dir')
        task_path = args_dict.pop('_task_path')
        dataset_names = args_dict['datasets']

        initialize_distributed(local_rank, world_size, rank=rank) # TODO arg
        for name in dataset_names:
            start_time = time.time()
            output_dir = os.path.join(task_path, name)
            # setup logs
            log_dir = os.path.join(output_dir, 'logs')
            set_logger(log_dir, rank=rank) # rank_{k}.log
            progress_record = ProgressRecord(log_dir, rank) # rank_{k}.progress
            # TODO check registered name when launch monitor
            # setup saver
            saver_cls = get_registered_cls(args_dict['saver_type'])
            saver = saver_cls(os.path.join(output_dir, name + saver_cls.suffix + f'.part_{rank}.cogdata'), **args_dict)
            # setup task
            task_cls = get_registered_cls(args_dict['task_type'])
            task = task_cls(saver=saver, **args_dict)
            transform_fn = task.get_transform_fn()
            # setup datasets
            with open(os.path.join(current_dir, name, 'cogdata_info.json'), 'r') as fin:
                ds_info = json.load(fin)
            assert ds_info['name'] == name
            ds_cls = get_registered_cls(ds_info['data_format']) # e.g. ZipDataset
            sub_datasets = [
                ds_cls(
                    path=os.path.join(current_dir, name, sub_dataset_name), 
                    world_size=world_size,
                    rank=rank,
                    transform_fn=transform_fn
                )
                for sub_dataset_name in ds_info['data_files']
            ]
            get_logger().debug(f'process {name}...')
            # others in ds_info are for dataset or task
            task.process(sub_datasets, progress_record=progress_record, 
                dataset_dir=os.path.join(current_dir, name),
                **args_dict, **ds_info)
            # if task forgot to record progress, set to 100% when finished
            end_time = time.time()
            speed = sum([len(ds) for ds in sub_datasets]) / (end_time - start_time)
            progress_record.update(100, 100, speed)