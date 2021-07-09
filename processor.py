import enum
from multiprocessing import Pool
from torch._C import device
from datasets import TarDataset, ZipDataset
from torchvision import transforms
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer.unified_tokenizer import get_tokenizer
from tokenizer.api import img2code
import argparse
import struct
from torchvision.transforms.functional import pil_to_tensor


def get_image_transforms(img_size=256):
    image_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        # transforms.ToTensor(),
        # transforms.Normalize([0.79093, 0.76271, 0.75340],
        #                     [0.30379, 0.32279, 0.32800])
    ])
    return image_transform


def read_text(txt_files, mode):
    text_dict = {}
    if mode == "json":
        import json
        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                t = json.load(fin)
                txt_list.extend(list(t.items()))
        tmp = []
        for k, v in tqdm(txt_list):
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
        for v in tqdm(txt_list):
            if 'cnShortText' not in v or len(v['cnShortText']) <= 1:
                print("warning: some item do not have cnShortText")
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
        for k, v in tqdm(txt_list):
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


def write(model, txts, imgs, index, output_dir="outputs/image_net"):
    codes = img2code(model, imgs).cpu().tolist()
    codes_byte = struct.pack(
        len(codes[0])*"i", *codes[0])
    for j in codes[1:]:
        codes_byte += struct.pack(
            len(j)*"i", *j)
    name = os.path.join(output_dir, str(os.getpid())+"_"+str(index))
    with open(name+".txt", "wb") as f:
        f.write(str.join("\n", txts).encode("utf-8"))
    with open(name+".bin", "wb") as f:
        f.write(codes_byte)


def extract_code(args, img_folders, text_dict, device, num_workers, worker_id, ratio=1):
    try:
        tokenizer = get_tokenizer(args)
        model = tokenizer.img_tokenizer.model
        datasets = []
        for img_folder in img_folders[:20]:
            if img_folder[-3:] == "rar":
                dataset = StreamingRarDataset
            elif img_folder[-3:] == "zip":
                dataset = ZipDataset
            elif img_folder[-3:] == "tar":
                dataset = TarDataset
            elif img_folder[-2:] == "h5":
                dataset = H5Dataset
            else:
                dataset = ImageFileDataset
            print(img_folder)
            dataset = dataset(img_folder, num_workers,
                              worker_id, get_image_transforms())
            datasets.append(dataset)
        print('Finish reading meta-data of dataset.')
        txt_mode = "dict"
        text_dict = read_text(txt_files, txt_mode)
        for dataset_index, dataset in enumerate(datasets):
            # loader = DataLoader(dataset, batch_size=32,
            #                    shuffle=False)
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
                if cnt > total_cnt * ratio:
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
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess args")
    parser.add_argument("--img_tokenizer_path", type=str,
                        default='vqvae_hard_biggerset_011.pt')
    parser.add_argument("--img_tokenizer_num_tokens", type=int, default=None)
    parser.add_argument("--encode_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_type", type=str, default="text_img")
    parser.add_argument("--ratio", type=float, default=1)
    args = parser.parse_args()
    print(args)

    image_list = os.listdir(
        "/dataset/fd5061f6/cogview/dingming0629/image_net/train/")
    img_folders = ["/dataset/fd5061f6/cogview/dingming0629/image_net/train/" +
                   image_path for image_path in image_list]

    txt_files = [
        "/dataset/fd5061f6/cogview/dingming0629/image_net/infolist.json"]
    ratio = args.ratio
    device = 'cuda'
    # pool = Pool(2)
    # for i in range(2):
    #     pool.apply_async(extract_code, args=(
    #         (args, img_folders, txt_files, device, 2, i)))
    # pool.close()
    # pool.join()
    extract_code(args, img_folders, txt_files, device, 1, 0)


def DataProcessor():
    def __init__(self, merge_dir):
        pass

    def updata_preprocess(self):
        pass

    def clear_preprocess(self):
        pass

    def _preprocess_all(self):
        pass

    def _preprocess_single_dataset(self, folder_name, config):
        pass

    def _make_text_img_batch(self, txts, imgs)
