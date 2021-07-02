from multiprocessing import Pool, Lock, Queue

from utils import get_data_list, format_file_size
import json
import shutil
import os
from torch.utils.data import Dataset, dataset


class DataManager():
    def __init__(self, config_path="defult_config.json") -> None:
        self.config = self.load_config(config_path)
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
        if os.path.isfile(path):
            print("Need directory!")
            return
        dir, name = os.path.split(path)
        if name in get_data_list(self.config['data_dir']) and dir != self.config['data_dir']:
            print("Name({}) exist!".format(name))
            return
        new_path = os.path.join(self.config['data_dir'], name)
        if os.path.isdir(path) and dir != self.config['data_dir']:
            shutil.move(path, new_path)
        self.info[name]['label'] = label
        self.info[name]['mode'] = mode
        self.info[name]['len'] = self.get_len(path, mode)
        if mode == "zip":
            self.info[name]['zip_file'] = new_path
        self.info.update(attributes)

    def get_len(self, path, mode):
        if mode == "rar":
            from unrar import rarfile
            rar = rarfile.RarFile(path)
            return len(rar.namelist())
        elif mode == "zip":
            from zipfile import ZipFile
            zip = ZipFile(path)
            return len([info for info in zip.infolist()
                        if info.filename[-1] != os.sep])
        else:
            return len(os.listdir(path))

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

    def random_split(self, name, k):
        datasets_len = self.info[name]['len']
        split_len = datasets_len//k
        cur = 0
        begin_idx = []
        while cur < datasets_len:
            begin_idx.append(cur)
            cur += split_len
        return datasets_len, begin_idx


class DataProcessor():
    def task_start(self, f,  input, output, path=None, k=1):
        pool = Pool(k)
        if path is None:
            for i in range(k):
                pool.apply_async(f, args=(input, output,))
        else:
            for i in range(k):
                pool.apply_async(f, args=(path, input, output,))
        return pool

    def task_join(self, pool):
        pool.close()
        pool.join()

    def run(self, manager, name,  inputs, outputs, workers, read, preprocess, write):
        mode = manager.info[name]['mode']
        if mode == "zip":
            input_path = manager.info[name]["zip_file"]
            read_pool = self.task_start(
                read, inputs[0], outputs[0], workers[0], input_path)
            read_pool.close()

            preprocess_pool = self.task_start(
                preprocess, inputs[1], outputs[1], workers[1])
            preprocess_pool.close()

            output_dir = os.join(manager.config["output_dir"], name)
            write_pool = self.task_start(
                write, inputs[2], outputs[2], workers[2], output_dir)
            write_pool.close()

            read_pool.join()
            preprocess_pool.join()
            write_pool.join()

            if "info" in outputs[3]:
                index = str.join("\t", outputs[3]["info"])
                info_path = os.path.join(output_dir, "index.bin")
                with open(info_path, "wb") as f:
                    f.write(index.encode("utf-9"))
                manager.info[name]['output_info'] = info_path


class CogDataSet(Dataset):
    def __init__(self, manager, names) -> None:
        super().__init__()
        self.manager = manager
        self.infos = []
        self.sums = []
        self.names = names
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
