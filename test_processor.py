#!/usr/bin/env python

from cogdata.utils.logger import set_logger, get_logger
set_logger('tmp')

if __name__ == '__main__':
    from cogdata.data_processor import DataProcessor
    import argparse
    import os
    print('cli')
    args_dict = {}
    image_list = os.listdir(
        "/dataset/fd5061f6/cogview/dingming0629/image_net/train/")
    args_dict['image_folders'] = ["/dataset/fd5061f6/cogview/dingming0629/image_net/train/" +
                                  image_path for image_path in image_list]

    args_dict['txt_files'] = [
        "/dataset/fd5061f6/cogview/dingming0629/image_net/infolist.json"]
    args_dict['task'] = "text_image"
    args_dict['img_tokenizer_path'] = "vqvae_hard_biggerset.pt"
    args_dict['device'] = 'cuda'
    proc = DataProcessor()
    proc.run_monitor(args_dict)

    # use argparse to finish cli
