from multiprocessing import Pool, Lock, Queue
import struct

from utils import get_data_list, format_file_size
import json
import shutil
import os
from torch.utils.data import Dataset


class DataManager():
    def __init__(self, config_path="defult_config.json") -> None:
        self.config = self.load_config(config_path)
        if "output_dir" not in self.config:
            self.config["output_dir"] = "outputs"
        self.info = self.load_info(self.config['info_path'])
        self.data_type = {}

    def load_info(self, info_path):
        info = {}
        if os.path.exists(info_path):
            with open(info_path) as fin:
                info = json.load(fin)
        return info

    def load_config(self, config_path):
        manage_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(manage_dir, 'default_config.json'), 'r') as fin:
            config = json.load(fin)

        if os.path.exists(config_path):
            with open(config_path, 'r') as fin:
                new_config = json.load(fin)
                config.update(new_config)
        return config

    def list_dataset(self):
        sizes = []
        data_dir = self.config['data_dir']
        for ds in self.info:
            size = 0
            if os.path.isdir(os.path.join(data_dir, ds)):
                for f in os.listdir(os.path.join(data_dir, ds)):
                    if os.path.isfile(os.path.join(data_dir, ds, f)):
                        size += os.path.getsize(os.path.join(data_dir, ds, f))
            else:
                size += os.path.getsize(os.path.join(data_dir, ds))
            sizes.append(size)
        s = sum(sizes)
        ret = list(zip(self.info, sizes))
        ret = sorted(ret, key=lambda x: x[1])
        for k, v in ret:
            print('{:20}{:20} {}'.format(
                k, self.info[k]['label'], format_file_size(v)))
        print('Total size: {}'.format((format_file_size(s))))

    def add_dataset(self, path, label, mode="single", attributes={}):
        if mode == "single" and os.path.isfile(path):
            print("Need directory!")
            return
        dir, name = os.path.split(path)
        if name in get_data_list(self.config['data_dir']) and dir != self.config['data_dir']:
            print("Name({}) exist!".format(name))
            return
        new_path = os.path.join(self.config['data_dir'], name)
        if os.path.isdir(path) and dir != self.config['data_dir']:
            shutil.move(path, new_path)
        self.info[name] = {"label": label, "mode": mode, "path": new_path}
        self.info[name].update(attributes)

    def set_dataset_label(self, name, new_label):
        self.data_labels[name] = new_label

    def rm_dataset_by_label(self, label):
        for i in self.info:
            if self.info[i]['label'] == label:
                del self.info[i]

    def rm_dataset_by_name(self, name):
        del self.info[name]

    def commit_info(self):
        with open(self.config['info_path'], "w") as fout:
            json.dump(self.info, fout)

    def get_name_by_label(self, label):
        ans = []
        for i in self.info:
            if self.info[i]['label'] == label:
                ans.append(i)
        return ans

    def get_label_by_name(self, name):
        return self.info[name]['label']




class CogDataSet(Dataset):
    def __init__(self, manager, names, img_size) -> None:
        super().__init__()
        self.manager = manager
        self.infos = []
        self.sums = []
        self.names = names
        self.img_size = img_size
        sum = 0
        for name in names:
            path = self.manager.info[name]["output_info"]
            with open(path, "rb") as f:
                info = f.read().decode("utf-8").strip().split("\t")
            self.infos.append(info)
            sum += int(info[-1])
            self.sums.append(sum)

    def __len__(self):
        return self.sums[-1]

    def __getitem__(self, index):
        dataset_idx = 0
        for sum in self.sums:
            if sum > index:
                break
            dataset_idx += 1

        def bisearch(l, x):
            lo = 0
            hi = len(l)-1
            while lo < hi:
                mi = (lo + hi)//2
                if x > int(l[mi]):
                    lo = mi + 1
                else:
                    hi = mi
            return hi

        file_index = bisearch(self.infos[dataset_idx], index)
        begin_idx = 0
        if file_index > 0:
            begin_idx = self.infos[file_index][-1]
        code_index = index - begin_idx

        file_name = os.path.join(
            self.manager.config['output_dir'], str(file_index)+".bin")
        with open(file_name, "rb") as f:
            code = f.read().decode("utf-8").strip().split("\t\t")[code_index]
        return code.strip().split("\t")
