import shutil
from utils import get_data_list, format_file_size
import json
import os
from torch.utils.data import Dataset


class DataManager():
    def __init__(self, merge_dir=None, config=None) -> None:
        """
        merge_dir: directory of merge result
        config: config.json in Merge_dir
        """
        self.merge_dir = merge_dir
        if config is None:
            config = {}
        with open(os.path.join(merge_dir, 'config.json'), "w") as f:
            json.dump(config, f)
        self.config = config
        self.info = {}

    def update_merge(self):
        pass

    def clear_merge(self):
        pass

    def add_data(self, data_path, data_name, config):
        """
        data_path: path to a new dataset
        data_name: name of the dataset
        """
        if data_name in get_data_list(self.config['data_dir']) and dir != self.config['data_dir']:
            print("Name({}) exist!".format(data_name))
            return
        new_dir_path = os.path.join(self.config['data_dir'], data_name)
        os.mkdir(new_dir_path)
        shutil.move(data_path, os.join(new_dir_path))
        self.info[data_name] = {"name": data_name,
                                "format": data_format, "text_format": text_format, "merged": False, "removed": Fase}.update(config)
        with open(os.path.join(self.config['src_dir'], data_name, "config.json"), "w") as fout:
            json.dump(self.info[data_name], fout)

    def remove_dataset(self, data_name):
        """
        data_name: name of a existed dataset
        """
        self.info[data_name]['removed'] = True
        with open(os.path.join(self.config['src_dir'], data_name, "config.json"), "w") as f:
            config = json.load(f)
            config['removed'] = True
            json.dump(config, f)

    def _merge_all(self):
        pass

    def _merge_single_dataset(self, folder_name, config):
        """
        folder_name(str): dataset's name of folder under src_dir
        config(dict): information of this dataset
        """
        pass

    def _get_data(self):
        pass

    def _data_split(self, ds):
        pass

    def list_dataset(self):
        sizes = []
        data_dir = self.config['data_dir']
        for ds in self.meta_info:
            size = 0
            if os.path.isdir(os.path.join(data_dir, ds)):
                for f in os.listdir(os.path.join(data_dir, ds)):
                    if os.path.isfile(os.path.join(data_dir, ds, f)):
                        size += os.path.getsize(os.path.join(data_dir, ds, f))
            else:
                size += os.path.getsize(os.path.join(data_dir, ds))
            sizes.append(size)
        s = sum(sizes)
        ret = list(zip(self.meta_info, sizes))
        ret = sorted(ret, key=lambda x: x[1])
        for k, v in ret:
            print('{:20}:{}'.format(
                k,  format_file_size(v)))
        print('Total size: {}'.format((format_file_size(s))))


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
            self.manager.config['merge_dir'], str(file_index)+".bin")
        with open(file_name, "rb") as f:
            code = f.read().decode("utf-8").strip().split("\t\t")[code_index]
        return code.strip().split("\t")
