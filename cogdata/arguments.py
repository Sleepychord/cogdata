"""This module defines functions for parsing console arguments.
"""
import argparse
import sys
import os
from cogdata.data_savers import BinarySaver
from cogdata.data_manager import DataManager
from cogdata.utils.logger import get_logger
from cogdata.utils.helpers import get_registered_cls, load_code


def need_datasets(p):
    p.add_argument('datasets', type=str, nargs='*',
                   help='dataset names, None means all possible datasets.')


def need_taskid(p, required=True):
    p.add_argument('--task_id', '-t', type=str,
                   help='id of the handling task.', required=required)

def get_args():
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--extra_code', type=str, default=None)
    known, args_list = py_parser.parse_known_args()
    if known.extra_code is not None:
        load_code(known.extra_code)

    parser = argparse.ArgumentParser(prog='cogdata')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("create_dataset", aliases=['data'])
    subparser.add_argument('name', type=str)
    subparser.add_argument('--description', type=str, default='',
                           help='description of the handling dataset.')
    subparser.add_argument('--data_files', type=str, nargs='+', default=[],
                           help='file names of the handling dataset.', required=True)
    subparser.add_argument('--data_format', type=str, default='',
                           help='format of data files.', required=True)
    subparser.add_argument('--text_files', type=str,
                           nargs='+', default=[], help='file name of the text.')
    subparser.add_argument('--text_format', type=str,
                           default='', help='format of the text file.')
    subparser.set_defaults(func=DataManager.new_dataset)
    # TODO unknow param save

    subparser = subparsers.add_parser("create_task", aliases=['task'])
    subparser.add_argument('task_id', type=str)
    subparser.add_argument('--description', type=str,
                           default='', help='description of the new task.')
    # TODO post check after import customized codes
    subparser.add_argument('--task_type', type=str, default=None,
                           help='type of the handling task.', required=True)
    subparser.add_argument('--saver_type', type=str,
                           default='binary', help='saver mode.', required=True)
    subparser.add_argument('--length_per_sample', type=int,
                           default=1089, help='data length of one sample (Bytes).')
    subparser.add_argument('--img_sizes', type=int, nargs='+',
                           default=[256], help='sizes of a pre-tokenized image.')
    subparser.add_argument('--txt_len', type=int, default=64,
                           help='length of text in one sample.')
    subparser.add_argument('--dtype', type=str, default='int32',
                           help='data type of samples.', choices=list(BinarySaver.mapping.keys()))
    subparser.set_defaults(func=DataManager.new_task)
    subparser.add_argument('--model_path', type=str, default='', help='model_path', required=False)

    # TODO how to customize?

    subparser = subparsers.add_parser("list")
    need_taskid(subparser, False)
    subparser.set_defaults(func=DataManager.list)

    subparser = subparsers.add_parser("process")
    need_datasets(subparser)
    need_taskid(subparser)
    subparser.add_argument('--nproc', type=int, default=2,
                           help='number of processes to launch.')
    subparser.add_argument('--dataloader_num_workers', type=int, default=2,
                           help='number of processes for dataloader per computational process.')
    subparser.add_argument('--batch_size', type=int, default=128,
                           help='batch size.')                        
    subparser.add_argument('--ratio', type=float, default=1,
                           help='ratio of data to process')
    # subparser.add_argument('--device', type=str, default='cuda')
    subparser.set_defaults(func=DataManager.process)

    subparser = subparsers.add_parser("merge")
    need_taskid(subparser)
    subparser.set_defaults(func=DataManager.merge)

    subparser = subparsers.add_parser("split")
    need_taskid(subparser)
    subparser.add_argument('n', type=int, default=8,
                           help='number of split pieces for the merge result.')
    subparser.set_defaults(func=DataManager.split)

    subparser = subparsers.add_parser("clean")
    need_taskid(subparser)
    subparser.set_defaults(func=DataManager.clean)
    # subparser.add_argument('--display_num', type=int, default=0, help='number of samples to randomly display')

    args = parser.parse_args(args_list)
    if not hasattr(args, 'func'):
        sys.stderr.write('error: at least select a subcommand. see help.\n')
        parser.print_help()
        return None
    get_logger().debug(args)
    
    if known.extra_code is not None:
        args.extra_code = known.extra_code

    try: # is this necessary? load current registered dict and make choices TODO
        post_check(args)
    except KeyError as e:
        return None
    return args

def post_check(args):
    type_args = ['data_format', 'saver_type', 'task_type']
    for tp in type_args:
        if hasattr(args, tp):
            try:
                _cls = get_registered_cls(getattr(args, tp))
            except KeyError as e:
                sys.stderr.write(f'error: {tp} "{getattr(args, tp)}" is not found -- it is not registered.\n')
                raise e     
