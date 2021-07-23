#!/usr/bin/env python

import argparse
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logger import set_logger, get_logger
from data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--args_dict', type=str, default="{}")
parser.add_argument("--local_rank", type=int, default=None, required=True)
args = parser.parse_args()

get_logger().debug(args.args_dict)
args_dict = json.loads(args.args_dict)  # Will return a dictionary

proc = DataProcessor()
proc.run_single(args.local_rank, args_dict)
