# Streaming data module
本模块旨在

# Data Progress Loading
1. support change the data ratio (add / remove dataset), cannot change num_workers * dp_rank

2. each dataset state dict should record all passed samples in all workers and data parallels
The state of item dataset is the last index and random seed, meaning that perm by this seed and the index samples.
The state of iterable dataset are current urls with the final key.

(name, dp_rank, worker_id,seed,current_url,final_key)
(name, dp_rank, worker_id,seed,index)

3. for iterable dataset, cannot evenly split, copy until full and use different seed

3. how to record? yield out every time and manually update?
