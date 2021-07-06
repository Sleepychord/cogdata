# import argparse

# import torch
# from tokenizer.api import img2code
# from tokenizer.unified_tokenizer import get_tokenizer
# import os
# from torchvision import transforms
# import zipfile
# import struct
# from PIL import Image
# a = "gdasgadg"
# b = list(range(256))


# a = (a+"\n").encode("utf-8")
# b = struct.pack("i"*len(b), *b)
# print(len(b))

# with open("test.txt", "wb") as f:
#     f.write(b+a)
# with open("test.txt", "rb") as f:
#     bytes = f.read()
# print(bytes[1024:])


# img_size = 256
# zip = zipfile.ZipFile(
#     "/dataset/fd5061f6/cogview/dingming0629/ali_white_picts_256.zip")
# members = [info for info in zip.infolist() if info.filename[-1] != os.sep]
# tot_len = 1000
# transform = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.CenterCrop(img_size),
#     transforms.ToTensor(),
#     transforms.Normalize([0.79093, 0.76271, 0.75340], [
#         0.30379, 0.32279, 0.32800])])
# img = transform(Image.open(zip.open(members[0])))
# imgs = torch.stack([img])
# print(imgs.shape)
# zip.close()


# parser = argparse.ArgumentParser(description="preprocess args")
# parser.add_argument("--img_tokenizer_path", type=str,
#                     default='vqvae_hard_biggerset_011.pt')
# parser.add_argument('--img_tokenizer_num_tokens', type=int, default=None)
# parser.add_argument("--encode_size", type=int, default=32)
# parser.add_argument("--device", type=int, default=0)
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--page_size", type=int, default=1024*1024)
# parser.add_argument("--read_num", type=int, default=1)
# parser.add_argument("--process_num", type=int, default=1)
# parser.add_argument("--write_num", type=int, default=1)

# args = parser.parse_args()
# tokenizer = get_tokenizer(args)
# model = tokenizer.img_tokenizer.model.cuda()
# codes = img2code(model, imgs.cuda()).cpu().numpy().shape
# print(codes)

import os
import zipfile
from multiprocessing import Pool, Process, Manager
from torchvision import transforms


from PIL import Image


def read(index):
    print("start", os.getpid())
    try:
        data_path = "/dataset/fd5061f6/cogview/dingming0629/ali_white_picts_256.zip"
        print("load", os.getpid())
        zip = zipfile.ZipFile(data_path)
        print("select", os.getpid())
        file_list = [info for info in zip.infolist() if info.filename[-1]
                     != os.sep]
        print("begin", os.getpid())
        tot_len = len(file_list)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.79093, 0.76271, 0.75340], [
                0.30379, 0.32279, 0.32800])])
        mylen = (tot_len+2)//2
        begin_idx = list(range(0, tot_len, mylen))[index]
        print_info = -100000
        print("???", os.getpid())
        print("{},{}".format(begin_idx, begin_idx+mylen))
        print(tot_len)
        for i in range(begin_idx, begin_idx+mylen):
            if i >= tot_len:
                break
            imgs = Image.open(zip.open(file_list[i]))
            # imgs = transform(imgs)
            dirs, filename = os.path.split(file_list[i].filename)
            if (i-begin_idx)-print_info >= 100000:
                print("pid{} read:{}/{}".format(os.getpid(), i-begin_idx, mylen))
                print_info = i-begin_idx
        zip.close()
        print("end", os.getpid())
    except Exception as e:
        print("error", os.getpid())
        print(e)


# read(0)

try:
    pool = Pool(2)
    for i in range(2):
        pool.apply_async(read, args=(i,))
    pool.close()
    pool.join()
except Exception as e:
    print(e)
