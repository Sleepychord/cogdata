# -*- encoding: utf-8 -*-

from cogdata.utils.register import register
import os
import PIL
import torch
import json
from torch.utils.data import DataLoader

from .base_task import BaseTask
from cogdata.utils.logger import set_logger, get_logger
import numpy as np

from icetk import icetk as tokenizer


def txt_collate_fn(x):
    return x

@register
class IcetkTextTask(BaseTask):
    '''handle tokenization
    '''

    def __init__(self, saver, **kwargs) -> None:
        """config saver
        """
        self.saver = saver

    def get_transform_fn(self, transform=None, **kwargs):

        return transform

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
        text_format = kwargs.get('text_format', 'raw_txt')
        batch_size = kwargs.get('batch_size', 128)
        num_workers = kwargs.get('dataloader_num_workers', 2)
        ratio = kwargs.get('ratio', 1)

        cnt, total_cnt = 0, sum([len(dataset) for dataset in sub_datasets])
        for dataset in sub_datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=txt_collate_fn, pin_memory=True)
            for batch_txts in loader:
                
                if dataset.use_bytes_as_length:
                    cnt += sum(len(x) for x in batch_txts)
                else:
                    cnt += len(batch_txts)  # may not full batch
                if cnt > total_cnt * ratio:
                    break
                
                ret = []
                for txt in batch_txts:
                    if len(txt) == 0:
                        continue
                    if text_format == 'raw_txt':
                        ret.extend(tokenizer.encode(txt))
                        ret.append(tokenizer['</s>']) # </s> TODO
                    else:
                        tokenized_line = process_special_line(txt, text_format)
                        if tokenized_line is not None:
                            ret.extend(tokenized_line)
                            ret.append(tokenizer['</s>']) # </s> TODO
                if len(ret) == 0:
                    continue            
                data = torch.tensor(ret)

                self.saver.save(data)
                if cnt // batch_size % 50 == 0:
                    # get_logger().info("progress {}/{}".format(cnt, total_cnt))
                    if progress_record is not None:
                        progress_record.update(cnt, total_cnt)
            self.saver.commit()
        return 0

def process_special_line(data, txt_format, 
                         pile_valid_set=['Pile-CC', "OpenWebText2", "Wikipedia (en)"]):
    if txt_format == 'jsonl': # default ["text"]
        data = json.loads(data)
        text = data.get("text", None)
        return tokenizer.encode(text)
    elif txt_format == 'ccnews':
        data = json.loads(data)
        text = ""
        title = data.get("title", None)
        description = data.get("description", None)
        maintext = data.get("maintext", None)
        if title:
            text += title.strip() + " "
        if description and (not maintext or not maintext.startswith(description)):
            text += description.strip() + " "
        if maintext:
            text += maintext
        if len(text) > 100:
            return tokenizer.encode(text)
        else:
            return None
    elif txt_format == 'baike':
        data = json.loads(data)
        text = data.get("title", "") + data.get("abstract", "") + data.get("content", "")
        if text:
            return tokenizer.encode(text)
        else:
            return None
    elif txt_format == 'zhihu':
        data = json.loads(data)
        ans_length = len(data.get("ans-content", ""))
        ans_up = data.get("ans-up-num", "")
        ans_up = int(ans_up) if ans_up else 0
        if ans_length > 100 or ans_up > 1000:
            qtitle = data["q_title"]
            qcontent = data["q-content"]
            if qcontent is None:
                qcontent = ""
            else:
                qcontent = "问题描述：" + qcontent
            # user = data.get("user-signature", "")
            # if len(user) > 0:
            #     user = "回答用户：" + user
            text = "问题：" + qtitle + qcontent + "回答：" + data["ans-content"]
            return tokenizer.encode(text)
        else:
            return None
    elif txt_format == 'zhidao':
        data = json.loads(data)
        if "title" not in data:
            return None
        qtitle = data["title"]
        qcontent = data.get("content", "")
        if len(qcontent) > 0:
            qcontent = "问题描述：" + qcontent
        qtitle = "问题：" + qtitle 
        text_all = qtitle +  qcontent + "回答："
        if "best_answer" in data:
            text = data["best_answer"]["content"]
            if len(text) > 10:
                text_all += text
        for answer in data.get("other_answers", []):
            text = answer["content"]
            if len(text) > 100:
                text_all += text
        return tokenizer.encode(text_all)
    elif txt_format == 'pile':
        data = json.loads(data)
        source = data["meta"].get("pile_set_name", None)
        if source not in pile_valid_set:
            return None
        text = data.get('text', '')
        return tokenizer.encode(text)

