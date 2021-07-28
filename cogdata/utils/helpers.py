import sys
import os
import json

def format_file_size(fileSize) :
    """Translate file size into B,KB,MB,GB

    Returns
    -------
    str
        a human-read size.
    """
    for count in ['Bytes','KB','MB','GB']:
        if fileSize > -1024.0 and fileSize < 1024.0:
            return "%3.1f%s" % (fileSize, count)
        fileSize /= 1024.0
    return "%3.1f%s" % (fileSize, 'TB')

def dir_size(path):
    """Sum all files' size in a directory
    
    Returns
    -------
    int
        The total size of all files in the directory.
    """
    return sum(d.stat().st_size for d in os.scandir(path) if d.is_file())

def get_last_line(filename):
    """ get last line of a file

    Parameters
    ----------
    filename:str
        file name

    Returns
    -------
    str or None
        last line or None for empty file
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

def exec_full(filepath):
    global_namespace = {
        "__file__": filepath,
        "__name__": "cogdata.extra_code",
    }
    with open(filepath, 'rb') as file:
        sys.path.append(os.path.split(filepath)[0])
        exec(compile(file.read(), filepath, 'exec'), global_namespace)

def load_code(path):
    if os.path.exists(path):
        exec_full(path)
    else:
        get_logger().error('The extra code file {} does not exist. Skipping.'.format(path))


from .register import ALLCLASSES
from cogdata.data_savers import *
from cogdata.datasets import *
from cogdata.tasks import * 

def get_registered_cls(name):
    return ALLCLASSES[name]
