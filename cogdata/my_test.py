import torch
from cogdata.data_savers.binary_saver import BinarySaver
from datasets import BinaryDataset
from tqdm import tqdm
from cogdata.utils.cogview import get_tokenizer


# from_path = "/dataset/fd5061f6/video_data/cogdata_video_new/cogdata_task_VideoText7Frame_2level/quanjing_merge_notgen/quanjing_merge_biaoqing_lifestyle_yundong.bin.cogdata"
# to_path = "/dataset/fd5061f6/video_data/image_cogdata/cogdata_task_4leveltokens/fake_quanjing_video/quanjing_biaoqing_yundong_lifestyle_4level_fake64_1in7test.cogdata"
# video_dataset  = BinaryDataset(from_path, 9024, dtype='int32')
# output_saver = BinarySaver(to_path, dtype='int32')

# print("From:", from_path)
# print("To:", to_path)

# buffer = torch.ones(5440) * (-1)

# cnt = -2
# for sample in tqdm(video_dataset):
#     if cnt % 7 == 0:
#         buffer[:64] = sample[:64]
#         buffer[64:320] = sample[576:832]
#         buffer[320:1344] = sample[3904:4928]
#         output_saver.save(buffer)
#         output_saver.commit()
#     cnt += 1
    
from torchvision.utils import save_image
import os
model_path = 'downloads/vqvae_hard_biggerset_011.pt'
test_path = "/dataset/fd5061f6/video_data/image_cogdata/cogdata_task_4leveltokens/fake_quanjing_video/quanjing_biaoqing_yundong_lifestyle_4level_fake64_1in7.cogdata"
test_dataset = BinaryDataset(test_path, 5440, dtype='int32')
test_dir = "tmp/fake-4leveltoken-frame/"
tokenizer = get_tokenizer(model_path)
print(len(test_dataset))
for index, sample in enumerate(test_dataset):
    # if index % 4 != 0:
    #     continue
    x = 0
    while sample[x] != -1 and x < 64:
        x+=1
    text = tokenizer.DecodeIds(sample[:x])[0][0]
    print(f"[TEXT] ", text)
    # imgs16 = [tokenizer.img_tokenizer.DecodeIds(sample[64+256*i:64+256*(i+1)].to('cuda')) for i in range(7)]
    # imgs16 = torch.cat(imgs16, dim=0)
    # save_image(imgs16, os.path.join(test_dir, f'{index}_{text}_s16.jpg'), normalize=True)
    
    # imgs32 = [tokenizer.img_tokenizer.DecodeIds(sample[64+1024*i:64+1024*(i+1)].to('cuda')) for i in range(24)]
    imgs32 = [tokenizer.img_tokenizer.DecodeIds(sample[64+256:64+256+1024].to('cuda'))]
    imgs32 = torch.cat(imgs32, dim=0)
    save_image(imgs32, os.path.join(test_dir, f'{index}_{text}_s32.jpg'), normalize=True)
    breakpoint()
    

    