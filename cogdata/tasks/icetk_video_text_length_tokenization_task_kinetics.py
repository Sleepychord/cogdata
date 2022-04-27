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
import numpy as np
from icetk import icetk as tokenizer


# use simple split instead of scene detection now
# For Kinetics

def video_collate_fn(data):
    videos, filenames, lengths = zip(*data)
    return videos, filenames, lengths

@register
class IcetkVideoTextLengthTokenizationKineticsTask(BaseTask):
    def __init__(self, saver, img_sizes, **kwargs) -> None:
        """config saver
        """
        self.saver = saver
        self.img_sizes = sorted(img_sizes, reverse=False) # 从小到大
        self.img_size = max(img_sizes)
        if 'frame_num_per_clip' in kwargs and kwargs['frame_num_per_clip'] > 0:
            self.frame_num_per_clip = kwargs['frame_num_per_clip']
        else:
            self.frame_num_per_clip = 5
        # if 'clip_length' in kwargs and kwargs['clip_length'] > 0:
        #     self.clip_length = kwargs['clip_length']
        #     self.full_length = False
        # else:
        #     self.clip_length = None # means max length, for AR mode
        #     self.full_length = True
        
        self.clip_length = kwargs['clip_length'] # 4 or 2 or 1, to accord with cogvideo
        self.frame_interval = kwargs['frame_interval'] #4 or 2 or 1, to accord with cogvideo
        self.full_length = False

        if 'threshold' in kwargs and kwargs['threshold'] > 0:
            self.threshold = kwargs['threshold']
        else:
            self.threshold = 30.0
        if 'max_clip_per_video' in kwargs and kwargs['max_clip_per_video'] > 0:
            self.max_clip_per_video = kwargs['max_clip_per_video']
        else:
            # tmp!!! for pytest
            # FIXME
            self.max_clip_per_video = 20
        get_logger().warning(f'frame_num_per_clip: {self.frame_num_per_clip} ')
        get_logger().warning(f'clip_length: {self.clip_length} ')
        get_logger().warning(f'max_clip_per_video: {self.max_clip_per_video} ')
        get_logger().warning(f'full_length: {self.full_length} ')
    
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
            # if Quanjing
            # filename = filename.split('.')[0]
            # if Kinetics
            taskname = full_filename.split('/')[-2]

            try:
                vr = VideoReader(fp)
                fps = vr.get_avg_fps()
                origin_frame_num = vr._num_frame
                # if self.clip_length is None: 
                #     total_time = int(origin_frame_num / fps)-1 # cut out begining and ending
                #     tmp_clip_length = min(total_time, 16)
                # else:
                #     tmp_clip_length = self.clip_length
                tmp_clip_length = self.clip_length

                # interval = int(fps * (tmp_clip_length / (self.frame_num_per_clip-1)))
                interval = self.frame_interval
                
                downsampled_frame_num = int((origin_frame_num+interval-1)/interval)
                # 采样正中间的
                sampled_clip_num = min((downsampled_frame_num // self.frame_num_per_clip), self.max_clip_per_video)
                sampled_idx_start = (downsampled_frame_num-sampled_clip_num*self.frame_num_per_clip)//2
                sampled_frames_idx = list(range(sampled_idx_start, interval*sampled_clip_num*self.frame_num_per_clip+sampled_idx_start, interval))
                sampled_frames = vr.get_batch(sampled_frames_idx).asnumpy()
                
                # simple split
                img_groups = []
                for blocki in range(sampled_clip_num):
                    imgs = [Image.fromarray(sampled_frames[blocki*self.frame_num_per_clip+framei]).convert('RGB') for framei in range(self.frame_num_per_clip)]
                    if local_transform is not None:
                        imgs = [local_transform(img) for img in imgs]
                    img_groups.append(imgs)
                
                # if Kinetics
                return img_groups, taskname, float(tmp_clip_length)
                # if others
                # return img_groups, filename, float(tmp_clip_length)
            except:
                get_logger().warning(f'Video {full_filename} is damaged.')
                return list(), "", 1.

        return transform_fn

    # def detect_cuts(self, frames):
    #     cut_list = [] #不包括头尾
    #     frame_num = len(frames)
    #     for fr in range(frame_num):
    #         if fr == 0:
    #             curr_hsv = list(cv2.split(cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)))
    #             cut_list = [0, ]
    #             continue
    #         last_hsv = curr_hsv
    #         curr_hsv = list(cv2.split(cv2.cvtColor(frames[fr], cv2.COLOR_BGR2HSV)))
    #         frame_score = self.calculate_frame_score(curr_hsv, last_hsv)
    #         if frame_score >= self.threshold:
    #             cut_list.append(fr)
    #     cut_list.append(frame_num)
    #     return cut_list

    # def calculate_frame_score(self, curr_hsv, last_hsv):
    #     delta_hsv = [0, 0, 0, 0]
    #     for i in range(3):
    #         num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
    #         curr_hsv[i] = curr_hsv[i].astype(np.int32)
    #         last_hsv[i] = last_hsv[i].astype(np.int32)
    #         delta_hsv[i] = np.sum(
    #             np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
            
    #     return sum(delta_hsv[0:3]) / 3.0

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
        task_text_dict = self.read_text(text_files, text_format)
        device = kwargs.get('device', 'cuda')
        batch_size = kwargs.get('batch_size', 16)
        num_workers = kwargs.get('dataloader_num_workers', 2)
        txt_len = kwargs.get('txt_len', 64)
        ratio = kwargs.get('ratio', 1)
        img_sizes = self.img_sizes
        
        buf_videos = [torch.zeros(self.frame_num_per_clip, 3, self.img_size,
                               self.img_size, device=device, dtype=torch.float) for _ in range(self.max_clip_per_video*batch_size)]  # buffer
        buf_txts = torch.zeros(self.max_clip_per_video*batch_size, txt_len,
                               device='cpu', dtype=torch.int) - 1
        
        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])
        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=video_collate_fn, pin_memory=True)
            
            for batch_videos, raw_filenames, clip_lengths in loader:
                if len(batch_videos) == 0:
                    continue
                batch_codes = []
                filenames = []
                code_n = 0
                for video_i, video in enumerate(batch_videos):
                    if raw_filenames[video_i] not in task_text_dict:
                        print("Skip! raw_filenames[video_i] = ", raw_filenames[video_i])
                        continue
                    for group in video:
                        code = torch.cat([pil_to_tensor(x).unsqueeze(0)
                                    for x in group])
                        buf_videos[code_n] = code.to(device)
                        filenames.append((task_text_dict[raw_filenames[video_i]], str(clip_lengths[video_i])+"秒"))
                        code_n += 1

                cnt += len(batch_videos)
                if cnt > total_cnt * ratio:
                    break

                if code_n == 0:
                    continue
                
                buf_txts.fill_(tokenizer['<pad>'])
                for i, filename in enumerate(filenames):
                    code_time = tokenizer.encode(filename[1])
                    code_txt = tokenizer.encode(filename[0])[:txt_len-len(code_time)-3] # at least 3 <pad> for separation
                    for j, c in enumerate(code_txt):
                        if c == tokenizer['<n>']:
                            code_txt[j] = tokenizer['<pad>']
                    if self.full_length:
                        pad_first = 0
                    else:
                        pad_first = 1
                    buf_txts[i, pad_first:len(code_time)+pad_first] = torch.tensor(
                    code_time, dtype=torch.int)
                    buf_txts[i, len(code_time)+pad_first] = tokenizer['<n>']
                    buf_txts[i, len(code_time)+1+pad_first:len(code_time)+1+len(code_txt)+pad_first] = torch.tensor(
                    code_txt, dtype=torch.int)

                codes_txt = buf_txts[:code_n]

                videos = []
                for i in range(code_n):
                    videos.append(buf_videos[i] / 255.)
                # videos = normfunc(buf_videos[:n] / 255.)
                codes_video = []
                for k in range(code_n):
                    video = []
                    for i, img_size in enumerate(img_sizes):
                        if img_size != self.img_size:
                            tmp_clip = torch.nn.functional.interpolate(
                            videos[k], (img_size, img_size), mode='bilinear')
                        else:
                            tmp_clip = videos[k]
                    codes_video.append(
                        tokenizer.encode(
                                image_torch=tmp_clip, image_size=None, compress_rate=8 # TODO
                            ).type(torch.IntTensor).view(1, -1)
                    )
                codes_video = torch.cat(codes_video, dim=0)
                data = torch.cat((codes_txt, codes_video), dim=1)
                self.saver.save(data)
                if cnt // batch_size % 2 == 0:
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
        if mode == 'json_ks':
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                txt_list.extend(t["RECORDS"])
            tmp = []
            for v in txt_list:
                tmp.append((v['rawname'], v['cnShortText']))
            text_dict = dict(tmp)
            return text_dict
        elif mode == "json_quanjing":
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                txt_list.extend(t["RECORDS"])
            tmp = []
            for v in txt_list:
                tmp.append((v['uniqueKey'], v['shortText'].replace(",", " ")))
            text_dict = dict(tmp)
            return text_dict
        elif mode == "json_quanjing_en_to_cn":
            import json
            txt_list = []
            for txt in txt_files:
                with open(txt, 'r') as fin:
                    t = json.load(fin)
                txt_list.extend(t["RECORDS"])
            tmp = []
            for v in txt_list:
                tmp.append((v['uniqueKey'], v['cnShortText'].replace(",", " ")))
            text_dict = dict(tmp)
            return text_dict
        else:
            return defaultdict(str)
