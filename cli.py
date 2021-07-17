#!/usr/bin/env python
from arguments import get_args

if __name__ == '__main__':
    print('cli')
    # use argparse to finish cli
    args = get_args()
    args.func(args)
    
