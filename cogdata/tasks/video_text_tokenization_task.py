from cogdata.utils.register import register
import os
import torch
import shutil
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from decord import VideoReader

import ffmpeg

from .base_task import BaseTask
from cogdata.data_savers import BinarySaver
from cogdata.utils.logger import set_logger, get_logger
from cogdata.utils.cogview import get_tokenizer
import numpy as np

def video_collate_fn(data):
    videos, filenames = zip(*data)
    return videos, filenames

@register
class VideoTextTokenizationTask(BaseTask):
    def __init__(self, saver, img_sizes, frame_num, **kwargs) -> None:
        """config saver
        """
        self.saver = saver
        self.img_sizes = sorted(img_sizes, reverse=True)
        self.img_size = max(img_sizes)
        self.frame_num = frame_num
        if 'time_slice' in kwargs and kwargs['time_slice'] > 0:
            self.time_slice = kwargs['time_slice']
        else:
            self.time_slice = 0
    
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
            dirs, filename = os.path.split(full_filename)
            filename = filename.split('.')[0]
            
            vr = VideoReader(fp)
            fps = vr.get_avg_fps()
            interval = int(fps * self.time_slice)
            max_frame_idx = vr._num_frame
            frames_idx = list(range(0, interval * self.frame_num, interval))
            for i in range(self.frame_num):
                if frames_idx[i] >= max_frame_idx:
                    frames_idx[i] = max_frame_idx - 1
            frames = vr.get_batch(frames_idx).asnumpy()

            imgs = []
            for i in range(frames.shape[0]):
                img = Image.fromarray(frames[i]).convert('RGB')
                if local_transform is not None:
                    img = local_transform(img)
                imgs.append(img)
            return imgs, filename


        # def transform_fn(fp, full_filename, *args, local_transform=transform):
        #     try:
        #         if fp is None:
        #             raise ValueError('')
        #         dirs, filename = os.path.split(full_filename)
        #         filename = filename.split('.')[0]
        #         os.makedirs(os.path.join('.cache', dirs), exist_ok=True)
        #         with open(os.path.join('.cache', full_filename), 'wb') as video_file:
        #             video_file.write(fp.read())
        #     except:
        #         raise ValueError('wrong')

        #     if self.time_slice == 0:
        #         pass
        #     else:
        #         os.system(
        #             f"ffmpeg -y \
        #             -ss 0 -i {os.path.join('.cache', full_filename)} \
        #             -r {self.time_slice} -t {self.frame_num} \
        #             -f image2 \
        #             {os.path.join('.cache', '%03d.png')}"
        #         )
        #     imgs = []
        #     for i in range(self.frame_num):
        #         img_path = os.path.join('.cache', '{:03d}.png'.format(i+1))
        #         img = Image.open(img_path).convert('RGB')
        #         if local_transform is not None:
        #             img = local_transform(img)
        #         imgs.append(img)
        #         os.remove(img_path)
        #     shutil.rmtree(os.path.join('.cache', dirs))
        #     return imgs, filename
        return transform_fn

    def process(self, sub_datasets, progress_record=None, dataset_dir='', **kwargs):
        """ Use cuda to process batch data from dataloader,
            save via Saver,
            report progress every 1/5000 ?
            final commit saver

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
        if not ('text_files' in kwargs and 'text_format' in kwargs):
            text_files = text_format = None
        else:
            text_files = [os.path.join(dataset_dir, tf)
                            for tf in kwargs['text_files']]
            text_format = kwargs['text_format']
        text_dict = self.read_text(text_files, text_format)
        device = kwargs.get('device', 'cuda')
        batch_size = kwargs.get('batch_size', 16)
        num_workers = kwargs.get('dataloader_num_workers', 2)
        txt_len = kwargs.get('txt_len', 64)
        ratio = kwargs.get('ratio', 1)
        img_sizes = self.img_sizes

        tokenizer = get_tokenizer(
            kwargs.get('model_path', 'downloads/vqvae_hard_biggerset_011.pt')
        )
        normfunc = transforms.Normalize([0.79093, 0.76271, 0.75340], [
                                        0.30379, 0.32279, 0.32800])
        
        buf_videos = [torch.zeros(self.frame_num, 3, self.img_size,
                               self.img_size, device=device, dtype=torch.float) for _ in range(batch_size)]  # buffer
        buf_txts = torch.zeros(batch_size, txt_len,
                               device='cpu', dtype=torch.int) - 1

        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])
        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=video_collate_fn, pin_memory=True)
            for batch_videos, raw_filenames in loader:
                batch_codes = []
                for video in batch_videos:
                    code = torch.cat([pil_to_tensor(x).unsqueeze(0)
                                for x in video])
                    batch_codes.append(code)

                cnt += len(raw_filenames)
                if cnt > total_cnt * ratio:
                    break
                filenames = []
                for i, filename in enumerate(raw_filenames):
                    if filename != "not_a_image" and filename in text_dict:
                        buf_videos[len(filenames)] = batch_codes[i].to(device)
                        filenames.append(filename)
                    else:
                        pass
                n = len(filenames)
                if n == 0:
                    continue

                buf_txts.fill_(-1)
                for i, filename in enumerate(filenames):
                    txt = text_dict[filename]
                    code_txt = tokenizer(txt)[:txt_len]
                    buf_txts[i, :len(code_txt)] = torch.tensor(
                        code_txt, dtype=torch.int)
                codes_txt = buf_txts[:n]

                videos = []
                for i in range(n):
                    videos.append(normfunc(buf_videos[i] / 255.))
                # videos = normfunc(buf_videos[:n] / 255.)
                codes_video = []
                for k in range(n):
                    video = []
                    for j in range(self.frame_num):
                        imgs = videos[k][j].unsqueeze(0)
                        for i, img_size in enumerate(img_sizes):
                            if i > 0:
                                tmp_imgs = torch.nn.functional.interpolate(
                                    imgs, (img_size, img_size), mode='bilinear')
                            else:
                                tmp_imgs = imgs
                            video.append(tokenizer.img_tokenizer.EncodeAsIds(
                                tmp_imgs).type(torch.IntTensor))
                    video = torch.cat(video, dim=1)
                    codes_video.append(video)
                codes_video = torch.cat(codes_video)
                data = torch.cat((codes_txt, codes_video), dim=1)

                self.saver.save(data)
                if cnt // batch_size % 20 == 0:
                    get_logger().info("progress {}/{}".format(cnt, total_cnt))
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
