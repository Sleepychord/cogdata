import sys
import os
import json


def cl(text, color=''):
    if color == 'red':
        return('\033[1;31;0m{}\033[0m'.format(text))
    elif color == 'blue':
        return('\033[1;34;0m{}\033[0m'.format(text))
    else:
        return(text)

def format_file_size(fileSize) :
    for count in ['Bytes','KB','MB','GB']:
        if fileSize > -1024.0 and fileSize < 1024.0:
            return "%3.1f %s" % (fileSize, count)
        fileSize /= 1024.0
    return "%3.1f %s" % (fileSize, 'TB')


def get_data_list(data_dir):
    r = []
    ignore = [".","..",".git","__pycache__"]
    for f in os.listdir(data_dir):
        if f in ignore:
            continue
        if os.path.isdir(os.path.join(data_dir,f)):
            r.append(f)
    return r

