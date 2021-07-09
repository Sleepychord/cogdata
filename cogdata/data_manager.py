# -*- encoding: utf-8 -*-

import os
import sys
import math
import random

class DataManager():
    def __init__(self, args) -> None:
        pass
        self.current_dir = ''
    
    def list(self):
        '''List all datasets in current dir.

        dataset1(233 MB) rar json processed(10MB)
        dataset2(10 GB) zip json_ks unprocessed
        ------------ Sumary -----------------
        current taskname: image_text_tokenization
        number(2) raw(10.23GB) processed(10MB)
        unprocessed: dataset2 
        '''
        pass

    def new_dataset(self, dataset_name):
        '''Create a dataset subfolder and a template (cogdata_info.json) in it.
        One should manually handle the data files.
        '''
        pass

    def new_task(self, args):
        '''create a cogdata_workspace subfolder and cogdata_config.json with configs in args.
        '''
        pass

    def process(self, args):
        '''process one or some (in args) unprocessed dataset (detected).
        '''
        pass

    def merge(self, args):
        '''merge all current processed datasets.
        '''
        pass
    def split(self, args):
        '''split the merged files into N parts.
        '''
        pass
    