# -*- encoding: utf-8 -*-

import os
import sys
import math
import random


def format_file_size(fileSize):
    for count in ['Bytes', 'KB', 'MB', 'GB']:
        if fileSize > -1024.0 and fileSize < 1024.0:
            return "%3.1f %s" % (fileSize, count)
        fileSize /= 1024.0
    return "%3.1f %s" % (fileSize, 'TB')


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
        sizes = []
        data_dir = self.config['data_dir']
        for ds in self.meta_info:
            size = 0
            if os.path.isdir(os.path.join(data_dir, ds)):
                for f in os.listdir(os.path.join(data_dir, ds)):
                    if os.path.isfile(os.path.join(data_dir, ds, f)):
                        size += os.path.getsize(os.path.join(data_dir, ds, f))
            else:
                size += os.path.getsize(os.path.join(data_dir, ds))
            sizes.append(size)
        s = sum(sizes)
        ret = list(zip(self.meta_info, sizes))
        ret = sorted(ret, key=lambda x: x[1])
        for k, v in ret:
            print('{:20}:{}'.format(
                k,  format_file_size(v)))
        print('Total size: {}'.format((format_file_size(s))))

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
