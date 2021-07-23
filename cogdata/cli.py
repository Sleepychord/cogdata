#!/usr/bin/env python
import sys

from cogdata.arguments import get_args

def main():
    args = get_args()
    if args is not None:
        args.func(args)

if __name__ == '__main__':
    main()
    
