import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default=None, help='user action')
    parser.add_argument('--base_dir', type=str, default=None, help='base directory.')

    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("task")
    subparser.add_argument('--task_id', type=str, default=None, help='id of the handling task.')
    subparser.add_argument('--task_type', type=str, default=None, help='type of the handling task.')
    subparser.add_argument('--saver', type=str, default='binary', help='saver mode.')
    subparser.add_argument('--length_per_sample', type=int, default=1089, help='data length of one sample (Bytes).')
    subparser.add_argument('--img_length', type=int, default=1024, help='length of a tokenized image.')
    subparser.add_argument('--txt_length', type=int, default=64, help='length of text in one sample.')
    subparser.add_argument('--dtype', type=str, default='int32', help='data type of samples.')
    subparser.add_argument('--split_num', type=int, default=1, help='number of split pieces for the merge result.')
    subparser.add_argument('--display_num', type=int, default=0, help='number of samples to randomly display')

    subparser = subparsers.add_parser("dataset")
    subparser.add_argument('--dataset', type=str, default=None, help='name of the handling dataset.')
    subparser.add_argument('--description', type=str, default='', help='description of the handling dataset.')
    subparser.add_argument('--data_files', type=list, default=[], help='file names of the handling dataset.')
    subparser.add_argument('--data_format', type=str, default=None, help='format of data files.')
    subparser.add_argument('--text_file', type=str, default=None, help='file name of the text.')
    subparser.add_argument('--text_format', type=str, default=None, help='format of the text file.')
    
    subparser = subparsers.add_parser("processor")
    subparser.add_argument('--local_rank', type=int, default=None, help='rank number for processor run_single')
    subparser.add_argument('--args_dict', type=str, default='{}')

    args = parser.parse_args()
    return args

'''
task level:
    python cli.py 
        --action add_task
        --task_type []
        --task_id []
        --saver []
        --length_per_sample []
        --dtype []

    python cli.py 
        --action process
        --task_id []
        --dataset [] （没有就处理所有未处理的）
    
    python cli.py 
        --action merge
        --task_id []

    python cli.py 
        --action split
        --task_id []
        --split_num []
    
dataset level:
    python cli.py 
        --action add_data
        --dataset []
        --description []
        --data_files []
        --data_format []
        --text_file []
        --text_format []
'''