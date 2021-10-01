from cogdata.utils.register import register
import os
import torch
import shutil
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from decord import VideoReader

# import ffmpeg
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
import cv2

from .base_task import BaseTask
from cogdata.data_savers import BinarySaver
from cogdata.utils.logger import set_logger, get_logger
from cogdata.utils.cogview import get_tokenizer
import numpy as np

def video_collate_fn(data):
    videos, filenames = zip(*data)
    return videos, filenames

@register
class VideoSceneTextTokenizationTask(BaseTask):
    def __init__(self, saver, img_sizes, **kwargs) -> None:
        """config saver
        """
        self.saver = saver
        self.img_sizes = sorted(img_sizes, reverse=True)
        self.img_size = max(img_sizes)
        if 'frame_num' in kwargs and kwargs['frame_num'] > 0:
            self.out_frame_num = kwargs['frame_num']
        else:
            self.out_frame_num = 16
        if 'interval' in kwargs and kwargs['sample_fps'] > 0:
            self.sample_fps = kwargs['sample_fps']
        else:
            self.sample_fps = 4
        if 'interval' in kwargs and kwargs['threshold'] > 0:
            self.threshold = kwargs['threshold']
        else:
            self.threshold = 45.0
        if 'max_clip_per_video' in kwargs and kwargs['max_clip_per_video'] > 0:
            self.max_clip_per_video = kwargs['max_clip_per_video']
        else:
            # tmp!!! for pytest
            # FIXME
            self.max_clip_per_video = 16
    
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

            # #detect scenes
            # video_manager = VideoManager([full_filename])
            # scene_manager = SceneManager()
            # scene_manager.add_detector(ContentDetector())
            # # Improve processing speed by downscaling before processing.
            # video_manager.set_downscale_factor()
            # # Start the video manager and perform the scene detection.
            # video_manager.start()
            # scene_manager.detect_scenes(frame_source=video_manager)

            # # Each returned scene is a tuple of the (start, end) timecode.
            # scene_timestamp_list = scene_manager.get_scene_list()
            try:
                vr = VideoReader(fp)
                fps = vr.get_avg_fps()
                interval = int(fps / self.sample_fps + 0.5)
                origin_frame_num = vr._num_frame
                sampled_frame_num = int((origin_frame_num+interval-1)/interval)
                sampled_frames_idx = list(range(0, interval * sampled_frame_num, interval))
                sampled_frames = vr.get_batch(sampled_frames_idx).asnumpy()
                
                # scene detect
                cut_list = self.detect_cuts(sampled_frames)
                img_groups = []
                clip_cnt = 0
                for cuti in range(len(cut_list)-1):
                    for blocki in range(int((cut_list[cuti+1]-cut_list[cuti])/self.out_frame_num)):
                        imgs = []
                        for framei in range(self.out_frame_num):
                            img = Image.fromarray(sampled_frames[cuti+blocki*self.out_frame_num+framei]).convert('RGB')
                            if local_transform is not None:
                                img = local_transform(img)
                            imgs.append(img)
                        img_groups.append(imgs)
                        clip_cnt += 1
                        if clip_cnt >= self.max_clip_per_video:
                            break
                    if clip_cnt >= self.max_clip_per_video:
                        break
                return img_groups, filename
            except:
                get_logger().warning(f'Video {full_filename} is damaged.')
                return list(), list()

        return transform_fn

    def detect_cuts(self, frames):
        cut_list = [] #不包括头尾
        frame_num = len(frames)
        for fr in range(frame_num):
            if fr == 0:
                curr_hsv = cv2.split(cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV))
                cut_list = [0, ]
                continue
            last_hsv = curr_hsv
            curr_hsv = cv2.split(cv2.cvtColor(frames[fr], cv2.COLOR_BGR2HSV))
            frame_score = self.calculate_frame_score(curr_hsv, last_hsv)
            if frame_score >= self.threshold:
                cut_list.append(fr)
        cut_list.append(frame_num)
        return cut_list

    def calculate_frame_score(self, curr_hsv, last_hsv):
        delta_hsv = [0, 0, 0, 0]
        for i in range(3):
            num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
            curr_hsv[i] = curr_hsv[i].astype(np.int32)
            last_hsv[i] = last_hsv[i].astype(np.int32)
            delta_hsv[i] = np.sum(
                np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
            
        return sum(delta_hsv[0:3]) / 3.0

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
        
        buf_videos = [torch.zeros(self.out_frame_num, 3, self.img_size,
                               self.img_size, device=device, dtype=torch.float) for _ in range(self.max_clip_per_video*batch_size)]  # buffer
        buf_txts = torch.zeros(self.max_clip_per_video*batch_size, txt_len,
                               device='cpu', dtype=torch.int) - 1

        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])
        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=video_collate_fn, pin_memory=True)
            for batch_videos, raw_filenames in loader:
                if len(batch_videos) == 0:
                    continue
                batch_codes = []
                batch_raw_filenames = []
                for video_i, video in enumerate(batch_videos):
                    for group in video:
                        code = torch.cat([pil_to_tensor(x).unsqueeze(0)
                                    for x in group])
                        batch_codes.append(code)
                        batch_raw_filenames.append(raw_filenames[video_i])

                cnt += len(raw_filenames)
                if cnt > total_cnt * ratio:
                    break
                filenames = []
                for i, filename in enumerate(batch_raw_filenames):
                    if filename != "failed_video" and filename in text_dict:
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
                    for i, img_size in enumerate(img_sizes):
                        for j in range(self.out_frame_num):
                            imgs = videos[k][j].unsqueeze(0)
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
                if cnt // batch_size % 1 == 0:
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
