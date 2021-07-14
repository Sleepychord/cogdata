# -*- encoding: utf-8 -*-
import time
import subprocess
import os
import torch
from torchvision import transforms

from cogdata.utils.cogview.unified_tokenizer import get_tokenizer
from cogdata.utils.logger import get_logger, set_logger
from cogdata.datasets import BinaryDataset,  TarDataset, ZipDataset
from cogdata.tasks.image_text_tokenization_task import ImageTextTokenizationTask


def _get_last_line(filename):
    """
    get last line of a file
    :param filename: file name
    :return: last line or None for empty file
    """
    try:
        filesize = os.path.getsize(filename)
        if filesize == 0:
            return None
        else:
            with open(filename, 'rb') as fp:  # to use seek from end, must use mode 'rb'
                offset = -8                 # initialize offset
                while -offset < filesize:   # offset cannot exceed file size
                    # read # offset chars from eof(represent by number '2')
                    fp.seek(offset, 2)
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:     # if contains at least 2 lines
                        # then last line is totally included
                        return lines[-1]
                    else:
                        offset *= 2         # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return lines[-1]
    except FileNotFoundError:
        print(filename + ' not found!')
        return None


class DataProcessor():
    def initialize_distributed(self, local_rank, args):
        """Initialize torch.distributed."""
        if local_rank is not None:
            device = local_rank
        torch.cuda.set_device(device)
        # Call the init process
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6005')
        init_method += master_ip + ':' + master_port
        get_logger().debug(init_method)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=args['world_size'], rank=local_rank,
            init_method=init_method)

    def __init__(self) -> None:
        pass

    def run_monitor(self, args_dict):
        '''Launch k run_single processes (by cmd, not multiprocess for dataloader)
           Monitor all the progresses by outputs in tmp files, clean tmp files from previous runs at first. use utils.progress_record !
           Wait and merge k files (use the helper in saver).
        '''
        set_logger(args_dict['log_dir'])
        command = ["python", "-m", "torch.distributed.launch", "--master_port", "30513", "--nproc_per_node",
                   str(args_dict['nproc_per_node']), "tests/test_processor.py", "--args_dict", str(args_dict).replace("\'", "\"")]
        p = subprocess.Popen(command)

        try:
            log_files = os.listdir(os.path.join(args_dict['log_dir'], "logs"))
            while p.poll() is None:
                for name in log_files:
                    if name == 'rank_main.log':
                        continue
                    last_line = _get_last_line(os.path.join(
                        os.getcwd(), args_dict['log_dir'], "logs", name))
                    get_logger().debug(last_line)
                time.sleep(1)
        except Exception as e:
            print(e)
            p.terminate()

    def run_single(self, local_rank, args_dict):
        '''really process, create datasets with task.transform_fn, iterating the dataloader and run task.process
        '''
        output_path = os.path.join(args_dict['output_dir'], "data.bin")
        if local_rank is not None:
            self.initialize_distributed(local_rank, args_dict)
            output_path = os.path.join(
                args_dict['output_dir'], str(local_rank)+".bin")
        set_logger(args_dict['log_dir'])
        image_folders = args_dict['image_folders']
        txt_files = args_dict['txt_files']
        task = args_dict['task']
        tokenizer = get_tokenizer(args_dict['img_tokenizer_path'])
        model = tokenizer.img_tokenizer.model
        args_dict['model'] = model
        datasets = []

        if task == "text_image":
            img_size = args_dict['img_size']
            task = ImageTextTokenizationTask(
                img_size, output_path)

            image_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                # transforms.ToTensor(),
                # transforms.Normalize([0.79093, 0.76271, 0.75340],
                #                     [0.30379, 0.32279, 0.32800])
            ])
            for img_folder in image_folders[:20]:
                if img_folder[-3:] == "rar":
                    dataset = StreamingRarDataset
                elif img_folder[-3:] == "zip":
                    dataset = ZipDataset
                elif img_folder[-3:] == "tar":
                    dataset = TarDataset
                else:
                    dataset = BinaryDataset
                dataset = dataset(
                    img_folder, task.get_transform_fn(image_transform))
                datasets.append(dataset)
            get_logger().debug('Finish reading meta-data of dataset.')
            txt_mode = "dict"
            text_dict = self.read_text(txt_files, txt_mode)
            args_dict['text_dict'] = text_dict
        for dataset_index, dataset in enumerate(datasets):
            get_logger().debug(
                '{} begin:{}/{}'.format(image_folders[dataset_index], dataset_index, len(datasets)))
            task.process(dataset_index, dataset, args_dict)
