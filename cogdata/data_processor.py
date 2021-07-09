# -*- encoding: utf-8 -*-

import os
import sys
import math
import random

class DataProcessor():
    def __init__(self, args) -> None:
        pass
    
    def run_monitor(self, args):
        '''Launch k run_single processes (by cmd, not multiprocess for dataloader)
           Monitor all the progresses by outputs in tmp files, clean tmp files from previous runs at first. use utils.progress_record !
           Wait and merge k files (use the helper in saver).
        '''
        pass
    
    def run_single(self, args):
        '''really process, create datasets with task.transform_fn, iterating the dataloader and run task.process
        '''
        pass