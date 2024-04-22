# usage: pytest tests/streaming/test_img.py
from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset
from transformers import AutoTokenizer

from torchvision.utils import save_image

def test_load_imgs(conf_path='tests/streaming/img_txt_testcase.yaml'):
    dataset = instantiate_from_yaml(conf_path, variables={
        'img_size': 224,
        'max_text_len': 256,
        'tokenizer_name': '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf'
    })
    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf',
        trust_remote_code=True,
        local_files_only=True
    )
    assert isinstance(dataset, MetaDistributedWebDataset)
    it = iter(dataset)
    for i in range(10):
        a = next(it)
        assert isinstance(a, dict)
        # save a['jpg'] (torch tensor)

        save_image(a['img'], f'tmp/test_img_{i}.jpg')
        print(i, tokenizer.decode(a['txt']))
        print(i, a['caption'])
        print(i, a['caption_zh'])
