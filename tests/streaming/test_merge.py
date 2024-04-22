from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset
from transformers import AutoTokenizer

def test_load_all(conf_path='tests/streaming/merge_testcase.yaml'):
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
    n = 200
    t0, t1 = 0, 0
    for i in range(n):
        a = next(it)
        if 'tokens' in a:
            t0 += 1
            # print('text.')
        else:
            t1 += 1
            # print('image.')
    assert abs(t0 - t1) < 0.05 * n

