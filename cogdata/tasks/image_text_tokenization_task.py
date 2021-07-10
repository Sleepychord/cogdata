# -*- encoding: utf-8 -*-

import os
import sys
import math
import random
from PIL import Image

from .base_task import BaseTask

class ImageTextTokenizationTask(BaseTask):
    '''handle tokenization
    '''
    def __init__(self) -> None:
        raise NotImplementedError
    
    def get_transform_fn(self, transform=None):
        '''
        Args:
            transform: a transform in torchvision, do not use ToTensor().
        '''
        def transform_fn(fp, full_filename, local_transform=transform):
            '''file obj to (PIL.Image, filename w/o suffix)
            '''
            img = Image.open(fp)
            dirs, filename = os.path.split(full_filename)
            filename = filename.split('.')[0]
            if local_transform is not None:
                img = local_transform(img)
        return transform_fn
