#!/usr/bin/env python

import argparse

from data_manager import DataManager
from arguments import get_args

def add_task(manager, args):
    pass

def add_data(manager, args):
    pass

def process(manager, args):
    pass

def merge(manager, args):
    pass

def split(manager, args):
    pass

if __name__ == '__main__':
    print('cli')
    # use argparse to finish cli
    args = get_args()
    base_dir = args.base_dir
    action = args.action

    manager = DataManager(base_dir)
    if action == 'add_task':
        add_task(manager, args)
    elif action == 'add_data':
        add_data(manager, args)
    elif action == 'process':
        process(manager, args)
    elif action == 'merge':
        merge(manager, args)
    elif action == 'split':
        split(manager, args)
    else:
        print("Please give a correct action.")
