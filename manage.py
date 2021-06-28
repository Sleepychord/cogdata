import sys
import os
import json

def init():
    global config
    global manage_dir
    global main_dir
    manage_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.getcwd()

    with open(os.path.join(manage_dir, 'default_config.json'), 'r') as fin:
        config = json.load(fin)
    
    config_path = os.path.join(main_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as fin:
            new_config = json.load(fin)
            config.update(new_config)

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

    
def get_data_list():
    global main_dir
    r = []
    for f in os.listdir(main_dir):
        if os.path.isdir(f) and f != '.' and f != '..':
            r.append(f)
    return r

def list_data():
    global main_dir
    r = get_data_list()
    sizes = []
    for ds in r:
        size = 0
        for f in os.listdir(os.path.join(main_dir, ds)):
            if os.path.isfile(os.path.join(main_dir, ds, f)):
                size += os.path.getsize(os.path.join(main_dir, ds, f))
        sizes.append(size)
    s = sum(sizes)
    ret = list(zip(r, sizes))
    ret = sorted(ret, key=lambda x: x[1])
    for k, v in ret:
        print('{}  {}'.format(k, format_file_size(v)))
    print('Total size: {}'.format((format_file_size(s))))

init()
list_data()
