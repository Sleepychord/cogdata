#!/usr/bin/env python

from cogdata.utils.logger import set_logger, get_logger
import argparse
import os
import json
set_logger('tmp')


parser = argparse.ArgumentParser()
parser.add_argument('--args_dict', type=str, default="{}")
parser.add_argument("--local_rank", type=int, default=None)
args = parser.parse_args()

get_logger().debug(args.args_dict)
args_dict = json.loads(args.args_dict)  # Will return a dictionary

image_list = os.listdir(
    "/dataset/fd5061f6/cogview/dingming0629/image_net/train/")
args_dict['image_folders'] = ["/dataset/fd5061f6/cogview/dingming0629/image_net/train/" +
                              image_path for image_path in image_list][0]
args_dict['txt_files'] = [
    "/dataset/fd5061f6/cogview/dingming0629/image_net/infolist.json"]
args_dict['task'] = "text_image"
args_dict['img_tokenizer_path'] = "vqvae_hard_biggerset.pt"
args_dict['device'] = 'cuda'
args_dict['world_size'] = 2
args_dict['nproc_per_node'] = 2

if args.local_rank == None:
    from cogdata.data_processor import DataProcessor
    proc = DataProcessor()
    proc.run_monitor(args_dict)
else:
    from cogdata.data_processor import DataProcessor
    proc = DataProcessor()
    proc.run_single(args.local_rank, args_dict)

# use argparse to finish cli
