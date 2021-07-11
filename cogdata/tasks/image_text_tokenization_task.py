# -*- encoding: utf-8 -*-

import os
import PIL
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
from PIL import Image

from .base_task import BaseTask
from ..utils.logger import get_logger
from ..data_savers import BinarySaver


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
                return Image.new('RGB', (self.img_size, self.img_size), (255, 255, 255)), None

            dirs, filename = os.path.split(full_filename)
            filename = filename.split('.')[0]
            if local_transform is not None:
                img = local_transform(img)
        return transform_fn

    def process(self, dataset_index, dataset, text_dict, args):
        saver = BinarySaver()
        device = args.device
        loader = dataset
        print(str(dataset) + " index: " + str(dataset_index))
        cnt = 0
        total_cnt = len(loader)
        raw_filenames = []
        raw_imgs = None
        normfunc = transforms.Normalize([0.79093, 0.76271, 0.75340], [
                                        0.30379, 0.32279, 0.32800])
        raw_imgs = torch.zeros(
            32, 3, 256, 256, device=device, dtype=torch.float)
        for raw_img, raw_filename in loader:
            raw_img = pil_to_tensor(raw_img)
            if len(raw_filenames) >= 32:
                raw_filenames = []
            raw_imgs[len(raw_filenames)] = raw_img.to(device)
            raw_filenames.append(raw_filename)
            if len(raw_filenames) < 32:
                continue
            raw_imgs = normfunc(raw_imgs)
            cnt += 1
            if cnt > total_cnt * args.ratio:
                break
            # imgs = []
            filenames = []
            filter_ids = []
            for i, filename in enumerate(raw_filenames):
                if filename != "not_a_image" and text_dict.__contains__(filename):
                    # imgs.append(raw_imgs[i])
                    filenames.append(filename)
                    filter_ids.append(i)
                else:
                    print("warning: deleted damaged image")
            # filtered_img = torch.stack(imgs)
            raw_imgs = raw_imgs[filter_ids]
            try:
                txts = [text_dict[filename] for filename in filenames]

                write(model, txts, raw_imgs, cnt)
            except KeyError:
                print("warning: KeyError. The text cannot be find")
                pass
            if cnt % 100 == 0:
                print("proc{}:{}/{}".format(os.getpid(), cnt, total_cnt))
