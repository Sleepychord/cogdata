# -*- encoding: utf-8 -*-
# @File    :   binary_saver.py
# @Time    :   2021/07/09 19:57:12
# @Author  :   Ming Ding
# @Contact :   dm18@mail.tsinghua.edu.cn

# here put the import lib
import os
import sys
import math
import random
import torch
from tqdm import tqdm
import numpy as np
from PIL.Image import Image
from .base_saver import BaseSaver
from cogdata.utils.register import register


@register
class CustomFrameSaver(BaseSaver):
    """Save data as binary files
    """

    def __init__(self, output_path, **kwargs):
        # TODO test speed of buffering
        self.save_root = output_path

    def save(self, datas, rawnames):
        '''
        Parameters
        ----------
        data:Tensor
            write in a binary file.
        '''
        for vidx, name in enumerate(rawnames):
            os.makedirs(os.path.join(self.save_root, name))
            for cidx, clip in enumerate(datas[vidx]):
                os.makedirs(os.path.join(self.save_root, name, str(cidx)))
                for fidx, frame in enumerate(clip):
                    fileidx = str(fidx).zfill(4)
                    Image.save(frame, fp=os.path.join(self.save_root, name, str(cidx), f'frame_{fileidx}.jpg'))

    def commit(self):
        '''
        Commit all buffered samples.
        '''
        pass

    def __del__(self):
        '''
        Close and Commit.
        '''
        pass

    # @classmethod
    # def merge(cls, files, output_path, overwrite=False):
    #     '''
    #     Merge files into one.
    #     '''
    #     if os.path.exists(output_path):
    #         if overwrite:
    #             os.remove(output_path)
    #         else:
    #             raise FileExistsError
    #     ret = os.system('cat {} >> {}'.format(
    #         ' '.join(files), output_path
    #     )) # TODO: solve possible "Argument list too long"
    #     if ret != 0:
    #         raise Exception(f'cat return code {ret}')

    @classmethod
    def merge(cls, files, output_path, overwrite=False):
        ''' Merge files into one.

        Parameters
        ----------
        files:[file pointer]
            Files which need to merge
        output_path:str
            Path of output file.
        '''
        merge_file = open(output_path, 'wb')

        for file_path in tqdm(files):
            with open(file_path, 'rb') as data_file:
                while True:
                    data = data_file.read(cls.max_buffer_size)
                    if not data:
                        break
                    merge_file.write(data)

        merge_file.close()

    # @classmethod
    # def split(cls, input_path, output_dir, n):
    #     '''
    #     Split input_path into n files in output_path.
    #     '''
    #     os.makedirs(output_dir, exist_ok=True)
    #     prefix = os.path.join(output_dir, os.path.split(input_path)[-1]+'.part')
    #     ret = os.system('split -d {} -n {} {} -a 3'.format(
    #         input_path, n, prefix
    #     ))
    #     if ret != 0:
    #         raise Exception(f'split return code {ret}')

