from cogdata.streaming import MergedDataset
from copy import deepcopy

def reshard_states(
        states, 
        dataset,
        new_dpsize, new_num_workers
        ):
    assert isinstance(dataset, MergedDataset)
    assert dataset.dp_size == 1
    ret = {}

    def dfs(x, prefix=''):
        if isinstance(x, MergedDataset):
            for sub_x in x.datasets:
                dfs(sub_x, prefix=prefix + f'f{x.name}.')
        else:
            ret[x.name] = x
    dfs(dataset)
    
    frontier = {k: -1 for k in ret.keys()}
    frontier_key = {}
    frontier_seed = {}
    for k, v in states.items():
        datasetname, dprank, workerid = k
        if len(v) == 3: # (seed, url, key)
            seed, url, key = v
            all_urls = ret[datasetname].urls
            assert isinstance(all_urls, list)
            url_index = all_urls.index(url)
            if url_index > frontier[datasetname]:
                frontier[datasetname] = url_index
                frontier_key[datasetname] = key
                frontier_seed[datasetname] = seed
        else: # (seed, index)
            seed, index = v
            if index > frontier[datasetname]:
                frontier[datasetname] = index
                frontier_key[datasetname] = None
                frontier_seed[datasetname] = seed
    # re shard
    new_state = {}
    for datasetname, v in frontier.items():
        if frontier_key[datasetname] is None:
            # item
            for dprank in range(new_dpsize):
                for workerid in range(new_num_workers):
                    new_state[(datasetname, dprank, workerid)] = (
                        frontier_seed[datasetname], v
                    )
        else: # iterable
            for dprank in range(new_dpsize):
                for workerid in range(new_num_workers):
                    # simulate the shard process

                    urls = deepcopy(ret[datasetname].urls)
                    front_url = urls[frontier[datasetname]]
                    for i in range(frontier[datasetname]):
                        urls[i] = None

                    if len(urls) < new_dpsize:
                        urls = urls * ((new_dpsize - 1) // len(urls) + 1)
                        assert len(urls) >= new_dpsize
                        # select the urls based on dp_rank
                    urls = urls[dprank::new_dpsize]
                    
                    # select the url based on workerid
                    if len(urls) < new_num_workers:
                        urls = urls * (new_num_workers // len(urls) + 1)
                        urls = urls[:new_num_workers]
                    urls = urls[workerid::new_num_workers]

                    # enumerate the urls to find the first untrained one
                    last_url = None
                    for u in urls:
                        if u is not None:
                            last_url = u
                            break
                        
                    new_state[(datasetname, dprank, workerid)] = (
                        None, last_url, None 
                        # ignore key because seed changes due to different dp
                    )
    return new_state
