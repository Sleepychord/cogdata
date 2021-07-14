# -*- encoding: utf-8 -*-

import os
import PIL
import torch
import struct
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision import transforms
from PIL import Image

from .base_task import BaseTask
from cogdata.utils.logger import get_logger
from cogdata.data_savers import BinarySaver
from cogdata.utils.logger import set_logger, get_logger
from cogdata.utils.cogview.api import img2code
import numpy as np


class ImageTextTokenizationTask(BaseTask):
    '''handle tokenization
    '''

    def __init__(self, img_size, output_path) -> None:
        self.saver = BinarySaver(output_path, "uint8")
        self.img_size = img_size

    def get_transform_fn(self, transform=None):
        '''
        Args:
            transform: a transform in torchvision, do not use ToTensor().
        '''
        def transform_fn(fp, full_filename, local_transform=transform):
            '''file obj to (PIL.Image, filename w/o suffix)
            '''
            try:
                if fp is None:
                    raise ValueError('')
                img = Image.open(fp).convert('RGB')
            except (OSError, PIL.UnidentifiedImageError, Image.DecompressionBombError, ValueError) as e:
                if not isinstance(e, ValueError):
                    get_logger().warning(f'Image {full_filename} is damaged.')
                return Image.new('RGB', (self.img_size, self.img_size), (255, 255, 255)), "not_a_image"
            dirs, filename = os.path.split(full_filename)
            filename = filename.split('.')[0]
            if local_transform is not None:
                img = local_transform(img)
            return img, filename
        return transform_fn

    def write(self, raw_filenames, raw_imgs, text_dict, args_dict):
        txts = []
        txt_len = args_dict['txt_len']
        for filename in raw_filenames:
            txt = text_dict[filename].encode("utf-8")
            if len(text_dict[filename]) < txt_len:
                txt = txt.ljust(txt_len, b'\0')
            else:
                txt = txt[:txt_len]
            txts.append(list(txt))
        raw_imgs = raw_imgs[:len(txts), :]
        txts = torch.tensor(txts)
        codes = img2code(args_dict['model'], raw_imgs).cpu()
        data = torch.cat((txts, codes), dim=1)
        self.saver.save(data)

    def process(self, dataset_index, dataset, args_dict):
        text_dict = args_dict['text_dict']
        device = args_dict['device']
        loader = dataset
        cnt = 0
        total_cnt = len(loader)
        raw_filenames = []
        raw_imgs = None
        normfunc = transforms.Normalize([0.79093, 0.76271, 0.75340], [
                                        0.30379, 0.32279, 0.32800])
        raw_imgs = torch.zeros(
            32, 3, 256, 256, device=device, dtype=torch.float)
        txt_len = args_dict['txt_len']
        for raw_img, raw_filename in loader:
            raw_img = pil_to_tensor(raw_img)
            if len(raw_filenames) >= 32:
                raw_filenames = []
            raw_imgs[len(raw_filenames)] = raw_img.to(device)
            raw_filenames.append(raw_filename)
            cnt += 1
            if len(raw_filenames) < 32:
                continue
            raw_imgs = normfunc(raw_imgs)
            if cnt > total_cnt * args_dict['ratio']:
                break
            # imgs = []
            filter_ids = []
            for i, filename in enumerate(raw_filenames):
                if filename != "not_a_image" and text_dict.__contains__(filename):
                    # imgs.append(raw_imgs[i])
                    filter_ids.append(i)
                else:
                    print("warning: deleted damaged image")
            # filtered_img = torch.stack(imgs)
            raw_imgs = raw_imgs[filter_ids]
            if len(filter_ids) < len(raw_filenames):
                raw_filenames.remove("not_a_image")
            try:
                self.write(raw_filenames, raw_imgs, text_dict, args_dict)
            except KeyError:
                get_logger().warn("warning: KeyError. The text cannot be find")
            get_logger().debug("{}/{}".format(cnt, total_cnt))
        if len(raw_filenames) > 0:
            self.write(raw_filenames, raw_imgs, text_dict, args_dict)
        self.saver.commit()
