from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from cogdata import DataManager, DataProcessor, CogDataSet
import zipfile
from tokenizer.unified_tokenizer import get_tokenizer
from tokenizer.api import img2code
import os
from multiprocessing import Queue, Lock
from torchvision import transforms

path = ""

manager = DataManager()
manager.add_data()


def read(data_path, input, output):
    input_queue = input['queue']
    output_queue = output['queue']
    text_dict = input["text_dict"]
    img_size = input['img_size']
    mode = input['mode']
    if mode == "zip":
        begin_idx = input_queue.get()
        zip = zipfile.ZipFile(data_path)
        namelist = zip.namelist()
        tot_len = len(namelist)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.79093, 0.76271, 0.75340], [
                0.30379, 0.32279, 0.32800])])
        for i in range(begin_idx, begin_idx+input['len']):
            if i >= tot_len:
                break
            filename = namelist[i]
            img = transform(zip.open(filename))
            txt = text_dict[filename]
            output_queue.put((img, txt))
        zip.close()
    else:
        pass
    lock = input['lock']
    lock.acquire()
    output['finish'] += 1
    lock.release()


def preprocess(input, output):
    input_queue = input['queue']
    output_queue = input['queue']
    read_k = input['k']
    args = input['args']
    batch_size = args['batch_size']
    tokenizer = get_tokenizer(args)
    model = tokenizer.img_tokenizer.model
    while True:
        imgs = []
        txts = []
        while True:
            try:
                img, txt = input_queue.get_nowait()
                imgs.append(img)
                txts.append(txt)
            except():
                if len(txts) > 0:
                    break
                if input['finish'] >= read_k:
                    lock = input['lock']
                    lock.acquire()
                    output['finish'] += 1
                    lock.release()
                    return
        s = imgs.shape[-1]
        assert s == imgs.shape[-2] == 256
        codes = img2code(model, imgs).cpu().numpy()
        for i in range(len(txts)):
            output_queue.put((txts[i], codes[i]))


def write(output_dir, input, output):
    input_queue = input['queue']
    process_k = input['k']
    buffer_size = input['buffer_size']
    lock = input['lock']
    buffer = ""
    cnt = 0
    while True:
        try:
            txt, code = input_queue.get_nowait()
            seq = txt+"\t"+code+"\t\t"
            if len(seq)+len(buffer) > buffer_size:
                with open(os.path.join(output_dir, str(input["index"])+".bin"), "wb") as f:
                    f.write(buffer.encode("utf-8"))
                lock.acquire()
                input["sum"] += cnt
                output["info"].append(str(input['sum']))
                input["index"] += 1
                lock.release()
                buffer = ""
                cnt = 0
            buffer += seq
            cnt += 1
        except():
            if input['finish'] >= process_k:
                if len(buffer) > 0:
                    lock.acquire()
                    with open(os.path.join(output_dir, str(input['index'])+".bin"), "wb") as f:
                        f.write(buffer.encode("utf-8"))
                    input["sum"] += cnt
                    output["info"].append(str(input['sum']))
                    lock.release()
                break


if __name__ == "__main__":
    manager = DataManager()
    processor = DataProcessor()

    read_lock = Lock()
    process_lock = Lock()
    write_lock = Lock()
    input_queue = Queue()
    read_process = Queue()
    process_write = Queue()
    name = ""
    img_size = 0
    args = {}
    batch_size = 0
    page_size = 0

    text_dict = {}
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

    workers = [1, 1, 1]

    mode = manager.info[name]['mode']

    read_input = {"queue": input_queue, "text_dict": text_dict,
                  "img_size": img_size, "lock": read_lock, "mode": mode}
    read_output = {"queue": read_process}

    process_input = {"queue": read_process, "args": args,
                     "batch_size": batch_size, "k": workers[0], "lock": process_lock}
    process_output = {"queue": process_write}

    index = []
    write_input = {"queue": read_process, "buffer_size": page_size,
                   "k": workers[1], "lock": write_lock}
    write_output = {}

    processor.run(manager, name, [read_input, process_input, write_input], [
                  read_output, process_output, write_output], workers, read, preprocess, write)

    dataset = CogDataSet(manager, [name])
    dataloader = DataLoader(dataset, batch_size=10)
    for i in dataloader:
        print(i)
