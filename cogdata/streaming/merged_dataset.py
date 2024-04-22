import torch
import os
import random

from torch.utils.data import IterableDataset, Dataset

from collections.abc import Generator
from typing import Dict, Tuple, List

def make_iterator_unstreaming(
    d: Dataset, 
    rank: int,
    world_size: int, 
    offset: int = 0, 
    shuffle: bool = True, 
    maxperm: int = 2000000,
    seed: int = 0
) -> Generator[Dict]:
    
    assert hasattr(d, '__getitem__') and hasattr(d, '__len__')
    
    perm_size = min(maxperm, len(d))
    perm = torch.randperm(perm_size, generator=torch.Generator().manual_seed(seed))
    
    start = offset - offset % world_size + rank
    if start <= offset:
        start += world_size
    for i in range(start, len(d), world_size):
        # every world_size * perm_size
        shuffled_index = (i - i % perm_size) + perm[(i % perm_size)] if shuffle else i
        data = d[shuffled_index]
        if isinstance(data, dict):
            data.update({'__index__': i, '__seed__': seed})
            yield data
        else:
            yield {'data': data, '__index__': i, '__seed__': seed}

def pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample

def make_iterator_streaming(
    d: IterableDataset, 
    offset_url: str = None,
    offset_key: str = None,
    shuffle: bool = True,
    bufsize: int = 1000,
    seed = 0
) -> Generator[Dict]:
    
    _last_url = offset_url
    offset_key_wrap = [offset_key] # to object
    def check_offset_key(x):
        '''Reload until offset_key. If found the key, set to None if found to skip this check.
        '''
        if offset_key_wrap[0] is None:
            return True
        else:
            # keep reading until the offset_key is found
            assert _last_url == offset_url, f'Error found in url skipping. Target URL: {offset_url}, Reading URL: {_last_url}'
            if x['__key__'] == offset_key_wrap[0]:
                offset_key_wrap[0] = None
            return False
        
    iterator = iter(d) if offset_url is None else d.__iter_from__(offset_url)
    if not shuffle:
        for sample in iterator:
            if check_offset_key(sample):
                yield sample
    
    buf = []
    rng = random.Random(seed)
    for sample in iterator:
        sample['__seed__'] = seed
        if _last_url is None or sample['__url__'] != _last_url:
            # pull out all the existing samples to make sure we don't mix up different URLs
            while len(buf) > 0:
                x = pick(buf, rng)
                if check_offset_key(x):
                    yield x
            rng = random.Random(seed) # reset RNG
            _last_url = sample['__url__']

        if len(buf) < bufsize:
            # fill the initial buffer
            buf.append(sample)
            continue
        else:
            # replace a sample in the buffer
            x = pick(buf, rng)
            buf.append(sample)
            if check_offset_key(x):
                yield x
    
    # pull out all the remaining samples
    while len(buf) > 0:
        x = pick(buf, rng)
        if check_offset_key(x):
            yield x

class MergedDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        datasets,
        cyclic=True,
        shuffle_unstreaming_subsets=True,
        shuffle_streaming_subsets=True,
        seed=0
    ):
        self.datasets = datasets
        self.cyclic = cyclic
        self.weights = []
        self.lengths = []
        self.dp_size = 1
        self.dp_rank = 0
        self.shuffle = shuffle_unstreaming_subsets
        self.online_shuffle = shuffle_streaming_subsets
        self.seed = seed
        self.dataloader_states = {}
        self.reload_from_url_level = False
            
        for i, d in enumerate(datasets):
            assert hasattr(d, 'name')
            assert hasattr(d, 'length') or \
                hasattr(d, '__len__') or \
                isinstance(d, MergedDataset), 'only support Webdataset, JsonlDataset, MergedDataset or Dataset with __len__().'
            assert hasattr(d, 'percent'), 'percent for {d.name} is required for MergedDataset.'

            self.weights += [1. / d.percent]
            if hasattr(d, '__len__'):
                x = len(d)
            elif hasattr(d, 'length'):
                x = d.length
            else:
                x = 0 # Merged dataset
            if hasattr(d, 'length_multiplier'):
                x *= d.length_multiplier
            if hasattr(d, 'length_divisor'):
                x /= d.length_divisor
            self.lengths.append(int(x))

    def set_data_parallel(self, dp_size, dp_rank):
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        for d in self.datasets:
            if hasattr(d, 'set_data_parallel'):
                d.set_data_parallel(dp_size, dp_rank)
            else: # not webdataset or merged
                assert hasattr(d, '__len__')
    
    def set_dataloader_states(self, 
            dataloader_states,
            reload_from_url_level=False
            ):
        self.dataloader_states = dataloader_states
        self.reload_from_url_level = reload_from_url_level
    
    def make_iterator(self, 
            d: Dataset, 
            dataloader_states: Dict[Tuple[str, int, int], Tuple] = {},
            seed_offset: int = 0
        ):
        if isinstance(d, MergedDataset):
            return d.__iter_from__(dataloader_states=dataloader_states)
        
        # get dprank and workerid
        dp_rank = self.dp_rank
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)

        if isinstance(d, torch.utils.data.IterableDataset):
            # already shard d by set_data_parallel and __iter__
            seed, offset_url, offset_key = dataloader_states.get((d.name, dp_rank, worker_id), (None, None, None))
            if seed is None:
                # if not reloading, assign a different seed for each data worker, in case url for webdataset if not enough and two workers have the same url.
                seed = self.seed + seed_offset + dp_rank * 64 + worker_id
            return make_iterator_streaming(
                d, offset_url, offset_key, shuffle=self.online_shuffle, seed=seed)
        else:
            # has __len__
            assert hasattr(d, '__len__')
            data_world_size = self.dp_size * num_workers
            data_rank = self.dp_rank * num_workers + worker_id

            seed, last_index = dataloader_states.get((d.name, dp_rank, worker_id), (self.seed + seed_offset, 0))
            
            return make_iterator_unstreaming(
                d, data_rank, data_world_size, 
                offset=last_index,
                shuffle=self.shuffle, seed=seed
            )
    
    def __iter__(self):
        return self.__iter_from__(
            dataloader_states=self.dataloader_states,
            reload_from_url_level=self.reload_from_url_level
            )
        
    def __iter_from__(self, dataloader_states, reload_from_url_level=False):
        iterators = []
        acc_weights = []
        epochs = []

        if reload_from_url_level:
            from copy import deepcopy
            dataloader_states_new = deepcopy(dataloader_states)
            for k, state in dataloader_states.items():
                if len(state) == 3: # streaming type
                    dataloader_states_new[k] = (state[0], state[1], None)
            dataloader_states = dataloader_states_new

        for d in self.datasets:
            iterators.append(self.make_iterator(d, dataloader_states))

        for i in range(len(self.datasets)):
            acc_weights.append(0)
            epochs.append(0)

        chosen = self.dp_rank % len(self.datasets) # init with different starting point to make it a bit randomness
        while True:
            acc_weights[chosen] += self.weights[chosen]
            if all([acc_weights[i] >= 1 for i in range(len(self.datasets))]):
                acc_weights = [x - 1 for x in acc_weights]
            if os.getenv('DEBUG', '0') == '1':
                print('Chosen:', chosen, 'type:', type(self.datasets[chosen]))
            try:
                x = next(iterators[chosen])
                if not isinstance(self.datasets[chosen], MergedDataset):
                    # the 1st level MergedDataset will create these meta info
                    x['__dprank__'] = self.dp_rank
                    x['__workerid__'] = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() is not None else 0
                    x['__datasetname__'] = self.datasets[chosen].name
                else:
                    x['__datasetname__'] = self.datasets[chosen].name + '.' + x['__datasetname__']
                yield x
            except StopIteration:
                if not self.cyclic: 
                    raise
                epochs[chosen] += 1
                iterators[chosen] = self.make_iterator(
                    self.datasets[chosen],
                    seed_offset=epochs[chosen] * 131
                    )
            
            chosen = min(range(len(self.datasets)), key=lambda i: acc_weights[i])

    
    def check_length(self, required_samples):
        len_pairs = []
        for i, d in enumerate(self.datasets):
            percent = 1. / self.weights[i] # percent
            if isinstance(d, MergedDataset):
                len_pairs.append(d.check_length(int(required_samples*percent)))
            else:
                l = self.lengths[i]
                len_pairs.append((
                    getattr(d, 'name', f'Anonymous'),
                    l, 
                    int(required_samples*percent)
                    ))
        return len_pairs
            
                
def mixed_collate(batch, possible_keys=None):
    # assert possible_keys is not None, 'keys in dict, some of them might be missing for some subdatasets. eg. ["image", "txt", "video"]'
    for sample in batch:
        assert isinstance(sample, dict)
    # collect keys
    if possible_keys is not None:
        for k in possible_keys:
            for sample in batch:
                if k not in sample:
                    sample[k] = None
    return batch

def to_state(sample):
    if isinstance(sample, dict):
        key = (sample['__datasetname__'],sample['__dprank__'], sample['__workerid__'])
        if '__url__' in sample:
            val = (sample['__seed__'], sample['__url__'], sample['__key__'])
        elif '__index__' in sample:
            val = (sample['__seed__'], sample['__index__'])
        else:
            raise ValueError('Unknown sample type')
    elif isinstance(sample, list):
        # union to_state(x)
        return {k: v for x in sample for k, v in to_state(x).items()}
    return {key: val}

def print_dataset_length_stats(info, indent="", print_header=True):
    if print_header:
        print('-------- Dataset Length Stats --------')
        print('\033[91m Red \033[0m Means the dataset is less than 1/4 of the required length')
        print('\033[32m Dark Green \033[0m Means the dataset is more than 2 times the required length')
        print('--------------------------------------')
    for item in info:
        # Check if the item is a tuple (datasetname, length, required_length)
        if isinstance(item, tuple):
            datasetname, length, required_length = item
            
            # Determine color based on length vs required_length
            color = "\033[0m"  # Default color
            if length < required_length / 4:
                color = "\033[91m"  # Red
            elif length > required_length * 2:
                color = "\033[32m"  # Dark Green
            
            print(f"{indent}{color}{datasetname}: Length = {length}, Required Length = {required_length}\033[0m")
        
        # If the item is a list, it's a subinfo
        elif isinstance(item, list):
            # print(f"{indent}")
            new_indent = indent + "    "
            print_dataset_length_stats(item, new_indent, print_header=False)
    if print_header:
        print('--------------------------------------')