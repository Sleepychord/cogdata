import sys
import os
import json

def format_file_size(fileSize) :
    for count in ['Bytes','KB','MB','GB']:
        if fileSize > -1024.0 and fileSize < 1024.0:
            return "%3.1f%s" % (fileSize, count)
        fileSize /= 1024.0
    return "%3.1f%s" % (fileSize, 'TB')

def dir_size(path):
    return sum(d.stat().st_size for d in os.scandir(path) if d.is_file())

def get_last_line(filename):
    """
    get last line of a file
    :param filename: file name
    :return: last line or None for empty file
    """
    try:
        filesize = os.path.getsize(filename)
        if filesize == 0:
            return None
        else:
            with open(filename, 'rb') as fp:  # to use seek from end, must use mode 'rb'
                offset = -8                 # initialize offset
                while -offset < filesize:   # offset cannot exceed file size
                    # read # offset chars from eof(represent by number '2')
                    fp.seek(offset, 2)
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:     # if contains at least 2 lines
                        # then last line is totally included
                        return lines[-1]
                    else:
                        offset *= 2         # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return lines[-1]
    except FileNotFoundError:
        return None


from .register import ALLCLASSES
from cogdata.data_savers import *
from cogdata.datasets import *
from cogdata.tasks import * 

def get_registered_cls(name):
    return ALLCLASSES[name]
