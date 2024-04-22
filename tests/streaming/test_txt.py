from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset
from transformers import AutoTokenizer

def test_load_jsonl(conf_path='tests/streaming/txt_en_testcase.yaml'):
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
    it = iter(dataset)
    for i in range(10):
        a = next(it)
        print(i, tokenizer.decode(a['tokens']))

    