from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset, to_state
from transformers import AutoTokenizer
from copy import deepcopy

def test_reload_item(conf_path='tests/streaming/img_jsonl_sft_testcase.yaml'):
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
    n = 500
    dataloader_states = {}
    for i in range(n):
        a = next(it)
        state = to_state(a)
        dataloader_states.update(state)
    saved_dataloader_states = deepcopy(dataloader_states)

    it2 = dataset.__iter_from__(saved_dataloader_states)
    for i in range(50):
        result = next(it)
        result2 = next(it2)
        cont = result.get('id')
        cont2 = result2.get('id')
        assert cont == cont2
    



    
    
