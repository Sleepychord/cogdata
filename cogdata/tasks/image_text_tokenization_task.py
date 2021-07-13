# -*- encoding: utf-8 -*-

import os
import sys
import math
import random
import PIL
from PIL import Image

from .base_task import BaseTask
from ..utils.logger import get_logger

class ImageTextTokenizationTask(BaseTask):
    '''handle tokenization
    '''
    def __init__(self, img_size) -> None:
        self.img_size = img_size
        raise NotImplementedError
    
    def get_transform_fn(self, transform=None):
        '''
        Args:
            transform: a transform in torchvision, do not use ToTensor().
        '''
        def transform_fn(fp, full_filename, *args, local_transform=transform):
            '''file obj to (PIL.Image, filename w/o suffix)
            '''
            try:
                if fp is None:
                    raise ValueError('')
                img = Image.open(fp).convert('RGB')
            except (OSError, PIL.UnidentifiedImageError, Image.DecompressionBombError, ValueError) as e:
                if not isinstance(e, ValueError):
                    get_logger().warning(f'Image {full_filename} is damaged.')
                return Image.new('RGB', (self.img_size, self.img_size), (255, 255, 255)), None

            dirs, filename = os.path.split(full_filename)
            filename = filename.split('.')[0]
            if local_transform is not None:
                img = local_transform(img)
        return transform_fn
