from typing import Any
from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset, to_state, mixed_collate
from transformers import AutoTokenizer
from copy import deepcopy
from torch.utils.data import DataLoader

def yield_fn(src):
    tmp = 0
    longs = []
    for x in src:
        longs.append(x['__datasetname__'])
        tmp += 1
        if tmp > 3:
            yield {'longs': longs}
            longs = []
            tmp = 0

class YieldFn():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return yield_fn(*args, **kwds)


def test_load_all(conf_path='tests/streaming/merge_testcase_yieldfn.yaml'):
    dataset = instantiate_from_yaml(conf_path, variables={
        'img_size': 224,
        'max_text_len': 256,
        'tokenizer_name': '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf'
    })
    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf',
        trust_remote_code=True,
        local_files_only=True
    )
    it = iter(dataset)
    n = 2
    for i in range(n):
        a = next(it)
        print(a)
        print('----------')
    print(to_state(a))