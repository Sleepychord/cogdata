# -*- encoding: utf-8 -*-
import os
import torch
from torchvision import transforms
import json

from cogdata.utils.cogview.unified_tokenizer import get_tokenizer
from cogdata.utils.logger import get_logger, set_logger
from cogdata.datasets import BinaryDataset,  TarDataset, ZipDataset
from cogdata.tasks.image_text_tokenization_task import ImageTextTokenizationTask


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

    def read_text(self, txt_files, mode):
        text_dict = {}
        if mode == "json":
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                    txt_list.extend(list(t.items()))
            tmp = []
            for k, v in txt_list:
                tmp.append((v['uniqueKey'], v['cnShortText']))
            text_dict = dict(tmp)
        elif mode == "txt":
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    lines = fin.readlines()
                for line in lines:
                    key, value = line[:-1].split('\t')
                    key = key[:-2]
                    txt_list.append((key, value))
            text_dict = dict(txt_list)
        elif mode == "json_ks":
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                txt_list.extend(t["RECORDS"])
            tmp = []
            for v in txt_list:
                if 'cnShortText' not in v or len(v['cnShortText']) <= 1:
                    get_logger().warn("warning: some item do not have cnShortText")
                    continue
                tmp.append((v['uniqueKey'], v['cnShortText']))
            text_dict = dict(tmp)
        elif mode == "tsv":
            import pandas as pd
            txt_list = []
            for txt in txt_files:
                t = pd.read_csv(txt, sep='\t')
                txt_list.extend(list(t.values))
            tmp = []
            for k, v in txt_list:
                tmp.append((str(k), v))
            text_dict = dict(tmp)
        elif mode == "dict":
            import json
            text_dict = {}
            for txt in txt_files:
                with open(txt, "r") as fin:
                    t = json.load(fin)
                    text_dict.update(t)
        return text_dict

    def __init__(self) -> None:
        pass

    def run_monitor(self, args_dict):
        '''Launch k run_single processes (by cmd, not multiprocess for dataloader)
           Monitor all the progresses by outputs in tmp files, clean tmp files from previous runs at first. use utils.progress_record !
           Wait and merge k files (use the helper in saver).
        '''
        command = "python -m torch.distributed.launch --master_port=30513 --nproc_per_node={} tests/test_processor.py --args_dict='{}'".format(
            args_dict['nproc_per_node'], str(args_dict).replace("\'", "\""))
        print(command)
        os.system(command)

    def run_single(self, local_rank, args_dict):
        '''really process, create datasets with task.transform_fn, iterating the dataloader and run task.process
        '''
        output_path = os.path.join(args_dict['output_dir'], "data.bin")
        set_logger("tmp")
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
