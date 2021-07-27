# -*- encoding: utf-8 -*-
# @File    :   tar_saver.py
# @Time    :   2021/07/13 20:26:06
# @Author  :   Ming Ding
# @Contact :   dm18@mail.tsinghua.edu.cn

# here put the import lib
import os
import sys
import math
import random
import time

import tarfile
from .base_saver import BaseSaver
from cogdata.utils.register import register


@register
class TarSaver(BaseSaver):
    """Save data as tar files
    """
    suffix = '.tar'

    def __init__(self, output_path, mode='w:', **kwargs):
        """Open a tar file

        Parameters
        ----------
        output_path:str
            The path of the result tar file
        mode:str
            open(output_path,mode)
        """
        self.tar = tarfile.open(output_path, mode, bufsize=16*1024*1024)  # 64M

    def save(self, fp, full_filename, file_size):
        '''
        Parameters
        ----------
        fp:file pointer
            Processed data files
        full_filename:str
            The name of ``fp`` in tar files
        '''
        tar_info = tarfile.TarInfo(name=full_filename)
        tar_info.size = file_size
        tar_info.mtime = time.time()
        self.tar.addfile(
            tarinfo=tar_info,
            fileobj=fp
        )

    def commit(self):
        pass  # TODO: not supported, how to commit?

    def __del__(self):
        '''
        Close and Commit.
        '''
        self.commit()
        self.tar.close()

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
        if os.path.exists(output_path):
            if overwrite:
                os.remove(output_path)
            else:
                raise FileExistsError
        ret = os.system('tar -n --concatenate --file={} {}'.format(
            output_path, ' '.join(files)
        ))  # TODO: solve possible "Argument list too long"
        if ret != 0:
            raise Exception(f'tar --concatenate return code {ret}')

    @classmethod
    def split(cls, input_path, output_dir, n):
        ''' Split input_path into n files in output_path.

        Parameters
        ----------
        input_path:str
            The path of the input tar file.
        output_dir:str
            The root folder of n output files.
        '''
        raise NotImplementedError  # not supported
