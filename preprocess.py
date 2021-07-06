import queue
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
import struct


def read_text(txt_files, txt_type):
    if txt_type == "json":
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
    elif manager.info[name]['txt_type'] == "tsv":
        import pandas as pd
        txt_list = []
        for txt in manager.info[name]['txt_files']:
            t = pd.read_csv(txt, sep='\t')
            txt_list.extend(list(t.values))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((str(k), v))
        return dict(tmp)


def read(input, output):
    input_queue = input['queue']
    output_queue = output['queue']
    img_size = input['img_size']
    data_path = input['path']
    mode = input['mode']
    text_dict = read_text(input['txt_files'], input['txt_type'])
    if mode == "zip":
        id = input_queue.get()
        read_k = input['k']
        zip = zipfile.ZipFile(data_path)
        file_list = [info for info in zip.infolist() if info.filename[-1]
                     != os.sep]
        tot_len = len(file_list)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.79093, 0.76271, 0.75340], [
                0.30379, 0.32279, 0.32800])])
        mylen = (tot_len+read_k)//read_k
        begin_idx = list(range(0, tot_len, mylen))[id]
        print_info = -10000
        for i in range(begin_idx, begin_idx+mylen):
            if i >= tot_len:
                break
            img = transform(Image.open(zip.open(file_list[i])))
            dirs, filename = os.path.split(file_list[i].filename)
            txt = text_dict[filename.split(".")[0]]
            output_queue.put((img, txt))
            if (i-begin_idx)-print_info >= 100:
                print("pid{} read:{}/{}".format(os.getpid(), i-begin_idx, mylen))
                print_info = i-begin_idx
        zip.close()
    lock = input['lock']
    lock.acquire()
    output['finish'].value += 1
    lock.release()


def preprocess(input, output):
    input_queue = input['queue']
    output_queue = output['queue']
    read_k = input['k']
    args = input['args']
    lock = input['lock']
    batch_size = 10
    lock.acquire()
    tokenizer = get_tokenizer(args)
    lock.release()
    model = tokenizer.img_tokenizer.model

    imgs = []
    txts = []
    flag = True
    while flag:
        try:
            img, txt = input_queue.get(timeout=1)
            imgs.append(img)
            txts.append(txt)
        except queue.Empty:
            if input['finish'].value >= read_k:
                flag = False
        if len(imgs) > batch_size or (len(imgs) > 0 and not flag):
            imgs = torch.stack(imgs).to(torch.cuda.current_device())
            codes = img2code(model, imgs).cpu().numpy()
            for i in range(len(txts)):
                output_queue.put((txts[i], codes[i]))
            txts = []
            imgs = []
    lock.acquire()
    output['finish'].value += 1
    lock.release()


def write(input, output):
    input_queue = input['queue']
    process_k = input['k']
    buffer_size = input['buffer_size']
    lock = input['lock']
    info = output['info']
    output_dir = input['path']
    buffer_txt = []
    codes = None
    seq_len = 0
    flag = True
    while flag:
        try:
            txt, code = input_queue.get(time_out=1)
            import struct
            code = struct.pack(len(code)*"i", *(code.tolist()))
            seq_len += len(txt)+len(code)+2
            if codes is None:
                codes = code
            else:
                codes += code
            buffer_txt.append(txt)
        except queue.Empty:
            if input['finish'].value >= process_k:
                flag = False
        if seq_len > buffer_size or (not flag and seq_len > 0):
            lock.acquire()
            index = input['index'].value
            input["sum"].value += len(buffer_txt)
            info.append(input['sum'].value)
            input["index"].value += 1
            lock.release()
            with open(os.path.join(output_dir, str(index)+".txt"), "wb") as f:
                f.write(str.join("\n", buffer_txt).encode("utf-8"))
            with open(os.path.join(output_dir, str(index)+".bin"), "wb") as f:
                f.write(codes)
            buffer_txt = []
            codes = None
            seq_len = 0


def run(manager, name,  inputs, outputs, workers, funcs):
    mode = manager.info[name]['mode']
    if mode == "zip":
        input_path = manager.info[name]["path"]
        dir_name = name.split(".")[0]
        output_dir = os.path.join(manager.config["output_dir"], dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pools = []
        inputs[0]['path'] = input_path
        inputs[-1]['path'] = output_dir

        for i in range(1):
            pool = Pool(workers[i])
            for i in range(workers[i]):
                pool.apply_async(funcs[i], args=(inputs[i], outputs[i]))
            pool.close()
            pools.append(pool)
        for i in pools:
            i.join()

        if "info" in outputs[2]:
            info = outputs[2]['info']
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
    parser.add_argument("--read_num", type=int, default=5)
    parser.add_argument("--process_num", type=int, default=5)
    parser.add_argument("--write_num", type=int, default=5)
    parser.add_argument('--img_tokenizer_num_tokens', type=int, default=None)
    parser.add_argument('--workers', nargs=3,
                        type=int, default=[1, 1, 1])
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
    read_process = m.Queue()
    process_write = m.Queue()
    read_finish = m.Value("i", 0)
    process_finish = m.Value("i", 0)
    sum = m.Value('i', 0)
    index = m.Value('i', 0)
    read_lock = m.Lock()
    process_lock = m.Lock()
    write_lock = m.Lock()
    info = m.list()

    for i in range(workers[0]):
        input_queue.put(i)

    read_input = {"queue": input_queue, "txt_files": manager.info[name]['txt_files'], "txt_type": manager.info[name]
                  ['txt_type'], "img_size": img_size, "lock": read_lock, "mode": mode, "k": workers[0]}
    read_output = {"queue": read_process, "finish": read_finish}

    process_input = {"queue": read_process, "args": args,
                     "batch_size": batch_size, "k": workers[0], "finish": read_finish, "lock": process_lock}
    process_output = {"queue": process_write, "finish": process_finish}

    write_input = {"queue": process_write, "buffer_size": page_size, "finish": 0,
                   "k": workers[1], "lock": write_lock, "finish": process_finish, "sum": sum, "index": index}
    write_output = {"info": info}

    print("running...")
    run(manager, name, [read_input, process_input, write_input], [
        read_output, process_output, write_output], workers, [read, preprocess, write])
    # processor.run(manager, name, [read_input, process_input, write_input], [
    #               read_output, process_output, write_output], workers, [read, preprocess, write])

    # dataset = CogDataSet(manager, [name])
    # dataloader = DataLoader(dataset, batch_size=10)
    # for i in dataloader:
    #     print(i)
