# -*- encoding: utf-8 -*-

from cogdata.utils.register import register
import os
import PIL
import torch

from torch.utils.data import DataLoader

from .base_task import BaseTask
from cogdata.utils.logger import set_logger, get_logger
from cogdata.utils.ice_tokenizer import get_tokenizer
import numpy as np

def txt_collate_fn(x):
    return x

@register
class BilingualTextTokenizationTask(BaseTask):
    '''handle tokenization
    '''

    def __init__(self, saver, **kwargs) -> None:
        """config saver
        """
        self.saver = saver

    def get_transform_fn(self, transform=None):
        '''

        Parameters
        ----------
        transform:torchvision.transforms
            a transform in torchvision, do not use ToTensor().

        Returns
        -------
        function
            A transform function for images
        '''
        return transform

    def process(self, sub_datasets, progress_record=None, dataset_dir='', **kwargs):
        """ Use cuda to process batch data from dataloader,
            save via Saver,
            report progress every 1/5000 ?
            final commit saver

        Parameters
        ----------
        sub_datasets:[Dataset]
            All datasets in processing list
        progress_record:ProgressBar
            The progress bar for this task
        dataset_dir:str
            The path of the dataset folder

        Returns
        -------
        int
            0 - Process successfully
        """

        batch_size = kwargs.get('batch_size', 128)
        num_workers = kwargs.get('dataloader_num_workers', 2)
        ratio = kwargs.get('ratio', 1)

        tokenizer = get_tokenizer(
            kwargs.get('model_path', 'downloads/vqvae_hard_biggerset_011.pt') # TODO
        )

        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])
        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=txt_collate_fn, pin_memory=True)
            for batch_txts in loader:
                
                cnt += len(batch_txts)  # may not full batch
                if cnt > total_cnt * ratio:
                    break
                
                ret = []
                for txt in batch_txts:
                    ret.extend(tokenizer.encode(txt))
                    ret.append(tokenizer.eos) # </s> TODO
                    
                data = torch.tensor(ret)

                self.saver.save(data)
                if cnt // batch_size % 50 == 0:
                    get_logger().info("progress {}/{}".format(cnt, total_cnt))
                    if progress_record is not None:
                        progress_record.update(cnt, total_cnt)
            self.saver.commit()
        return 0
