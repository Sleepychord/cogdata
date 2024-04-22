import torch
import json
import re
import os
import random

from .web_dataset import DataPipeline, SimpleShardList, ReloadMixin
from torch.utils.data import IterableDataset, get_worker_info

import webdataset
from webdataset.handlers import reraise_exception
from webdataset.tariterators import url_opener
from webdataset.filters import pipelinefilter
from braceexpand import braceexpand

class JsonlIterableDataset(DataPipeline, ReloadMixin):
    def __init__(self, path, process_fn, *, shuffle_buffer=1000):

        # parse path, may mixed with dir
        # if there is a comma not between {}, add one for expansion
        path_wo_brace = re.sub(r"\{.*?\}", "", path)
        if ',' in path_wo_brace:
            path = '{' + path + '}'
        expanded_path = []
        for p in braceexpand(path):
            if p.endswith('.jsonl'):
                expanded_path.append(p)
            else:
                # assert a existing folder
                assert os.path.isdir(p), f"{p} is not a valid folder"
                # find all jsonl files
                for root, dirs, files in os.walk(p):
                    for file in files:
                        if file.endswith('.jsonl'):
                            file_path = os.path.join(root, file)
                            expanded_path.append(file_path)
        path = expanded_path

        super().__init__(
            SimpleShardList(path), # Lots of shards are recommended, or not evenly
            jsonl_samples,
            process_fn
        )

    def set_data_parallel(self, dp_size, dp_rank):
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.pipeline[0].set_data_parallel(dp_size, dp_rank)


def jsonl_expander(streams):
    for source in streams:
        for lineno, line in enumerate(source['stream']):
            sample = json.loads(line)
            sample['__url__'] = source['url']
            sample['__key__'] = str(lineno)
            for k,v in sample.items():
                if v is None:
                    sample[k] = ''
            yield sample

def jsonl_samples(src, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    return jsonl_expander(streams)


    