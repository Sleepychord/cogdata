import queue
import struct
import torch
from tqdm import tqdm
from cogdata import DataManager
import zipfile
from tokenizer.unified_tokenizer import get_tokenizer
from tokenizer.api import img2code
import os
from multiprocessing import Manager, Pool
from torchvision import transforms
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def process(input, output):
    print("read begin")
    try:
        input_queue = input['queue']
        img_size = input['img_size']
        data_path = input['in_path']
        output_dir = input['out_path']
        buffer_size = input['buffer_size']
        info = output['info']
        mode = input['mode']
        text_dict = input['text_dict']
        args = input['args']
        lock = input['lock']
        read_k = input['k']

        lock.acquire()
        tokenizer = get_tokenizer(args)
        lock.release()
        model = tokenizer.img_tokenizer.model

        if mode == "zip":
            id = input_queue.get()
            zip = zipfile.ZipFile(data_path)
            file_list = [info for info in zip.infolist(
            ) if info.filename[-1] != os.sep]
            tot_len = len(file_list)
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])])
            batch_size = buffer_size//4096*4
            mylen = (tot_len+read_k)//read_k
            begin_idx = list(range(0, tot_len, mylen))[id]
            print_info = -10000
            imgs = []
            txts = []
            for i in range(begin_idx, begin_idx+mylen):
                # if len(txts) > batch_size:
                #     lock.acquire()
                #     index = input['index'].value
                #     input["sum"].value += len(txts)
                #     info.append(input['sum'].value)
                #     input["index"].value += 1
                #     lock.release()
                #     imgs = torch.stack(imgs).cuda()
                #     codes = img2code(model, imgs).cpu().numpy()
                #     codes_byte = struct.pack(
                #         len(codes[0])*"i", *(codes[0].tolist()))
                #     for i in codes[1:]:
                #         codes_byte += struct.pack(
                #             len(i)*"i", *(i.tolist()))
                #     with open(os.path.join(output_dir, str(index)+".txt"), "wb") as f:
                #         f.write(str.join("\n", txts).encode("utf-8"))
                #     with open(os.path.join(output_dir, str(index)+".bin"), "wb") as f:
                #         f.write(codes_byte)
                #     txts = []
                #     imgs = []
                if i > tot_len:
                    break
                # imgs.append(transform(transforms.ToTensor()(
                #     Image.open(zip.open(file_list[i]))).cuda()))
                img = Image.open(zip.open(file_list[i]))
                img = transforms.functional.pil_to_tensor(img).float().cuda()
                img = transform(img)
                imgs.append(img)
                dirs, filename = os.path.split(file_list[i].filename)
                txts.append(text_dict[filename.split(".")[0]])
                if (i-begin_idx)-print_info >= 1000:
                    print("pid{} finish:{}/{}".format(os.getpid(), i-begin_idx, mylen))
                    print_info = i-begin_idx
            zip.close()
            if len(txts) > 0:
                lock.acquire()
                index = input['index'].value
                input["sum"].value += len(txts)
                info.append(input['sum'].value)
                lock.release()
                imgs = torch.stack(imgs).cuda()
                codes = img2code(model, imgs).cpu().numpy()
                codes_byte = struct.pack(
                    len(codes[0])*"i", *(codes[0].tolist()))
                for i in codes[1:]:
                    codes_byte += struct.pack(
                        len(i)*"i", *(i.tolist()))
                with open(os.path.join(output_dir, str(index)+".txt"), "wb") as f:
                    f.write(str.join("\n", txts).encode("utf-8"))
                with open(os.path.join(output_dir, str(index)+".bin"), "wb") as f:
                    f.write(codes_byte)

        print("finish")
    except Exception as e:
        print("error")
        print(e)


def run(manager, name,  input, output, workers, process):
    mode = manager.info[name]['mode']
    if mode == "zip":
        input_path = manager.info[name]["path"]
        dir_name = name.split(".")[0]
        output_dir = os.path.join(manager.config["output_dir"], dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        input['in_path'] = input_path
        input['out_path'] = output_dir

        pool = Pool(workers)
        for i in range(workers):
            pool.apply_async(process, args=(input, output))
        pool.close()
        pool.join()

        if "info" in output:
            info = output['info']
            info = struct.pack(len(info)*"i", *info)
            info_path = os.path.join(output_dir, "index.bin")
            with open(info_path, "wb") as f:
                f.write(info)
            manager.info[name]['output_info'] = info_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="preprocess args")
    parser.add_argument("--img_tokenizer_path", type=str,
                        default='vqvae_hard_biggerset_011.pt')
    parser.add_argument("--encode_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--page_size", type=int, default=1024*1024)
    parser.add_argument('--img_tokenizer_num_tokens', type=int, default=None)
    parser.add_argument('--workers',  type=int, default=1)
    args = parser.parse_args()
    print(args)

    manager = DataManager()

    manager.add_dataset("/dataset/fd5061f6/cogview/dingming0629/ali_white_picts_256.zip", "test", mode="zip", attributes={
                        "txt_files": ["/dataset/fd5061f6/cogview/dingming0629/sq_gouhou_white_pict_title_word_256_fulltitle.tsv"], "txt_type": "tsv"})
    img_size = args.encode_size * 8
    batch_size = args.batch_size
    page_size = args.page_size
    workers = args.workers
    name = manager.get_name_by_label("test")[0]
    mode = manager.info[name]['mode']

    m = Manager()
    input_queue = m.Queue()
    sum = m.Value('i', 0)
    index = m.Value('i', 0)
    read_lock = m.Lock()
    info = m.list()

    if manager.info[name]['txt_type'] == "json":
        import json
        txt_list = []
        for txt in manager.info[name]['txt_files']:
            with open(txt, 'r') as fin:
                t = json.load(fin)
                txt_list.extend(list(t.items()))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((v['uniqueKey'], v['cnShortText']))
        text_dict = dict(tmp)
    elif manager.info[name]['txt_type'] == "tsv":
        import pandas as pd
        txt_list = []
        for txt in manager.info[name]['txt_files']:
            t = pd.read_csv(txt, sep='\t')
            txt_list.extend(list(t.values))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((str(k), v))
        text_dict = dict(tmp)

    for i in range(workers):
        input_queue.put(i)

    input = {"queue": input_queue, "text_dict": text_dict, "args": args, "buffer_size": page_size,
             "img_size": img_size, "lock": read_lock, "mode": mode, "k": workers, "sum": sum, "index": index}
    output = {"info": info}

    print("running...")
    run(manager, name, input, output, workers, process)
    # processor.run(manager, name, [read_input, process_input, write_input], [
    #               read_output, process_output, write_output], workers, [read, preprocess, write])

    # dataset = CogDataSet(manager, [name])
    # dataloader = DataLoader(dataset, batch_size=10)
    # for i in dataloader:
    #     print(i)
