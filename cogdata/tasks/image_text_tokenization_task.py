# -*- encoding: utf-8 -*-

from cogdata.utils.register import register
import os
import PIL
import torch
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from .base_task import BaseTask
from cogdata.data_savers import BinarySaver
from cogdata.utils.logger import set_logger, get_logger
from cogdata.utils.cogview import get_tokenizer
import numpy as np


def img_collate_fn(data):
    imgs, filenames = zip(*data)
    return imgs, filenames


@register
class ImageTextTokenizationTask(BaseTask):
    '''handle tokenization
    '''

    def __init__(self, saver, img_sizes, **kwargs) -> None:
        self.saver = saver
        self.img_sizes = sorted(img_sizes, reverse=True)  # multi-scale
        self.img_size = max(img_sizes)

    def get_transform_fn(self, transform=None):
        '''

        Parameters
        ----------
        transform:torchvision.transforms
            a transform in torchvision, do not use ToTensor().

        Returns
        -------
        function
            A transform function for images
        '''
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size)
            ]
            )

        def transform_fn(fp, full_filename, *args, local_transform=transform):
            '''file obj to (PIL.Image, filename w/o suffix)

            Parameters
            ----------
                fp:
                    file pointer of the image file
                full_filename:str
                    file name with extension name
                local_transform:torchvision.transforms 
                    A transform for the image

            Returns
            -------
                img:torch.Tensor
                    The transformed result of the image
                filename:str
                    file name without extension name
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

    def process(self, sub_datasets, progress_record=None, dataset_dir='', **kwargs):
        """Process all datasets

        Parameters
        ----------
        sub_datasets:[Dataset]
            All datasets in processing list
        progress_record:ProgressBar
            The progress bar for this task
        dataset_dir:str
            The path of the dataset folder

        Returns
        -------
        int
            0 - Process successfully
        """
        # Get text dict {filename:text}
        if not ('text_files' in kwargs and 'text_format' in kwargs):
            text_files = text_format = None
        else:
            text_files = [os.path.join(dataset_dir, tf)
                          for tf in kwargs['text_files']]
            text_format = kwargs['text_format']
        text_dict = self.read_text(text_files, text_format)

        device = kwargs.get('device', 'cuda')
        batch_size = kwargs.get('batch_size', 128)
        num_workers = kwargs.get('dataloader_num_workers', 2)
        txt_len = kwargs.get('txt_len', 64)
        ratio = kwargs.get('ratio', 1)
        img_sizes = self.img_sizes

        tokenizer = get_tokenizer(
            kwargs.get('model_path', 'downloads/vqvae_hard_biggerset_011.pt')
        )
        normfunc = transforms.Normalize([0.79093, 0.76271, 0.75340], [
                                        0.30379, 0.32279, 0.32800])
        # for vqvae_hard_011.pt
        buf_imgs = torch.zeros(batch_size, 3, self.img_size,
                               self.img_size, device=device, dtype=torch.float)  # buffer
        buf_txts = torch.zeros(batch_size, txt_len,
                               device='cpu', dtype=torch.int) - 1

        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])

        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=img_collate_fn, pin_memory=True)

            for batch_imgs, raw_filenames in loader:
                batch_imgs = [pil_to_tensor(x)
                              for x in batch_imgs]  # TODO test speed

                cnt += len(raw_filenames)  # may not full batch
                if cnt > total_cnt * ratio:
                    break
                filenames = []
                for i, filename in enumerate(raw_filenames):
                    if filename != "not_a_image" and filename in text_dict:
                        buf_imgs[len(filenames)] = batch_imgs[i].to(
                            device)  # TODO test pack to
                        filenames.append(filename)
                    else:
                        get_logger().warning(f"deleted 1 damaged image.")
                n = len(filenames)  # valid num
                if n == 0:
                    continue

                buf_txts.fill_(-1)
                for i, filename in enumerate(filenames):
                    txt = text_dict[filename]
                    code_txt = tokenizer(txt)[:txt_len]
                    buf_txts[i, :len(code_txt)] = torch.tensor(
                        code_txt, dtype=torch.int)
                codes_txt = buf_txts[:n]

                imgs = normfunc(buf_imgs[:n] / 255.)
                codes_img = []
                for i, img_size in enumerate(img_sizes):
                    if i > 0:
                        tmp_imgs = torch.nn.functional.interpolate(
                            imgs, (img_size, img_size), mode='bilinear')
                    else:
                        tmp_imgs = imgs
                    codes_img.append(tokenizer.img_tokenizer.EncodeAsIds(
                        tmp_imgs).type(torch.IntTensor))
                data = torch.cat((codes_txt, *codes_img), dim=1)

                self.saver.save(data)
                if cnt // batch_size % 20 == 0:
                    get_logger().info("rank{}/{}".format(cnt, total_cnt))
                    if progress_record is not None:
                        progress_record.update(cnt, total_cnt)
            self.saver.commit()
        return 0

    def read_text(self, txt_files, mode):
        """Read text dict from text files

        Parameters
        ----------
        txt_files:[str]
            All names of the text files
        mode:str
            The mode of the text, including json,txt,json_ks,tsv,dict
        """
        from collections import defaultdict
        if txt_files is None:  # no txt, accept all
            return defaultdict(str)
        text_dict = {}
        if mode == "json":
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                    txt_list.extend(list(t.items()))
            tmp = []
            for k, v in txt_list:
                tmp.append((v['uniqueKey'], v['cnShortText']))
            text_dict = dict(tmp)
        elif mode == "txt":
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    lines = fin.readlines()
                for line in lines:
                    key, value = line[:-1].split('\t')
                    key = key[:-2]
                    txt_list.append((key, value))
            text_dict = dict(txt_list)
        elif mode == "json_ks":
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                txt_list.extend(t["RECORDS"])
            tmp = []
            for v in txt_list:
                if 'cnShortText' not in v or len(v['cnShortText']) <= 1:
                    get_logger().warning("some item do not have cnShortText")
                    continue
                tmp.append((v['uniqueKey'], v['cnShortText']))
            text_dict = dict(tmp)
        elif mode == "tsv":
            import pandas as pd
            txt_list = []
            for txt in txt_files:
                t = pd.read_csv(txt, sep='\t')
                txt_list.extend(list(t.values))
            tmp = []
            for k, v in txt_list:
                tmp.append((str(k), v))
            text_dict = dict(tmp)
        elif mode == "dict":
            import json
            text_dict = {}
            for txt in txt_files:
                with open(txt, "r") as fin:
                    t = json.load(fin)
                    text_dict.update(t)
        return text_dict
