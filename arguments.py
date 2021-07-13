import argparse

def add_task_args(parser):
    parser.add_argument('--task_id', type=str, default=None, help='id of the handling task.')
    parser.add_argument('--task_type', type=str, default=None, help='type of the handling task.')
    parser.add_argument('--saver', type=str, default='binary', help='saver mode.')
    parser.add_argument('--length_per_sample', type=int, default=1089, help='data length of one sample (Bytes).')
    parser.add_argument('--dtype', type=str, default='int32', help='data type of samples.')
    parser.add_argument('--split_num', type=int, default=1, help='number of split pieces for the merge result.')
    parser.add_argument('--display_num', type=int, default=0, help='number of samples to randomly display')

    return parser

def add_dataset_args(parser):
    parser.add_argument('--dataset', type=str, default=None, help='name of the handling dataset.')
    parser.add_argument('--description', type=str, default='', help='description of the handling dataset.')
    parser.add_argument('--data_files', type=list, default=[], help='file names of the handling dataset.')
    parser.add_argument('--data_format', type=str, default=None, help='format of data files.')
    parser.add_argument('--text_file', type=str, default=None, help='file name of the text.')
    parser.add_argument('--text_format', type=str, default=None, help='format of the text file.')
    
    return parser

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str, default=None, help='user action')
    parser.add_argument('--base_dir', type=str, default=None, help='base directory.')
    
    parser = add_task_args(parser)
    parser = add_dataset_args(parser)

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