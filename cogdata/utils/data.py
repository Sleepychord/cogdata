import sys
import os
import json

def format_file_size(fileSize) :
    for count in ['Bytes','KB','MB','GB']:
        if fileSize > -1024.0 and fileSize < 1024.0:
            return "%3.1f %s" % (fileSize, count)
        fileSize /= 1024.0
    return "%3.1f %s" % (fileSize, 'TB'), 