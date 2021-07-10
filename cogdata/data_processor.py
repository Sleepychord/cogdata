# -*- encoding: utf-8 -*-

import os
import sys
import math
import random


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


class DataProcessor():
    def __init__(self, args) -> None:
        pass

    def run_monitor(self, args):
        '''Launch k run_single processes (by cmd, not multiprocess for dataloader)
           Monitor all the progresses by outputs in tmp files, clean tmp files from previous runs at first. use utils.progress_record !
           Wait and merge k files (use the helper in saver).
        '''
        from multiprocessing import Pool
        pool = Pool(args.num_workers)
        for i in range(2):
            pool.apply_async(self.run_single, args=(
                (args)))
        pool.close()
        pool.join()

    def run_single(self, args):
        '''really process, create datasets with task.transform_fn, iterating the dataloader and run task.process
        '''
        image_list = os.listdir(args.image_dir)
        img_folders = ["/dataset/fd5061f6/cogview/dingming0629/image_net/train/" +
                       image_path for image_path in image_list]

        txt_files = args.txt_files
        ratio = args.ratio
        device = args.device
        extract_code(args, img_folders, txt_files, device, 1, 0)
