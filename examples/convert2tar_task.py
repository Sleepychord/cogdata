# -*- encoding: utf-8 -*-
'''
@File    :   convert2tar_task.py
@Time    :   2021/07/28 23:25:09
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

from cogdata.utils.register import register
from cogdata.tasks import BaseTask
from torch.utils.data import DataLoader

from convert2tar_helpers import file_collate_fn


@register
class Convert2TarTask(BaseTask):
    '''handle conversion
    '''

    def __init__(self, saver, **kwargs) -> None:
        """config saver
        """
        self.saver = saver
        
    def get_transform_fn(self, transform=None):
        return None

    def process(self, sub_datasets, progress_record=None, dataset_dir='', **kwargs):
        if not ('text_files' in kwargs and 'text_format' in kwargs):
            assert False
            text_files = text_format = None
        else:
            text_files = [os.path.join(dataset_dir, tf)
                          for tf in kwargs['text_files']]
            text_format = kwargs['text_format']
        text_dict = self.read_text(text_files, text_format)

        batch_size = kwargs.get('batch_size', 128)
        num_workers = kwargs.get('dataloader_num_workers', 2)
        ratio = kwargs.get('ratio', 1)
        
        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])

        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=file_collate_fn, pin_memory=False)

            for batch in loader:
                cnt += len(batch)  # may not full batch
                if cnt > total_cnt * ratio:
                    break
                for fp, filename, filesize in batch:
                    if filename in text_dict:
                        self.saver.save(fp, filename, filesize)
                if cnt // batch_size % 20 == 0:
                    if progress_record is not None:
                        progress_record.update(cnt, total_cnt)
            self.saver.commit()
        return 0
