import argparse
import os
import sys
import torch

sys.path.append('./')

from cogdata.utils.logger import set_logger, get_logger


def initialize_distributed(args):
    """Initialize torch.distributed."""
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.local_rank,
        init_method=init_method)


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=None)
parser.add_argument("--world_size", type=int, default=None)
args = parser.parse_args()

print(sys.argv)
if args.local_rank is not None:
    initialize_distributed(args)
else: # attention ! to avoid recursively launching
    os.system('python -m torch.distributed.launch --nproc_per_node=2 tests/manual_test_logger.py --world_size=2 > test.txt 2>&1')

print(torch.distributed.is_initialized())
set_logger('tmp')
get_logger().debug('debug')
get_logger().warning('warning')
