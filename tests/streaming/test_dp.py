from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset, to_state
from transformers import AutoTokenizer
from copy import deepcopy

def test_reload_dp(conf_path='tests/streaming/merge_testcase.yaml'):
    dataset = instantiate_from_yaml(conf_path, variables={
        'img_size': 224,
        'max_text_len': 256,
        'tokenizer_name': '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf'
    })
    dataset2 = instantiate_from_yaml(conf_path, variables={
        'img_size': 224,
        'max_text_len': 256,
        'tokenizer_name': '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf'
    })
    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/shared/official_pretrains/hf_home/Llama-2-7b-hf',
        trust_remote_code=True,
        local_files_only=True
    )
    dataset.set_data_parallel(8, 0)
    dataset2.set_data_parallel(8, 5)
    it = iter(dataset)
    it_2 = iter(dataset2)
    n = 1024 + 512
    dataloader_states = {}
    dataloader_states2 = {}
    for i in range(n):
        a = next(it)
        b = next(it_2)
        state = to_state(a)
        state2 = to_state(b)
        dataloader_states.update(state)
        dataloader_states2.update(state2)
        assert list(state.values())[0] != list(state2.values())[0]
    saved_dataloader_states = deepcopy(dataloader_states)
    saved_dataloader_states2 = deepcopy(dataloader_states2)
    print('reloading')
    it2 = dataset2.__iter_from__(saved_dataloader_states2)
    # actuall some different due to acc_weights, select a circulating point
    for i in range(50):
        result = next(it_2)
        result2 = next(it2)
        if result.get('id') is not None:
            cont = result['id']
            cont2 = result2['id']
        else:
            cont = result.get('tokens', result.get('txt')).sum()
            cont2 = result2.get('tokens', result2.get('txt')).sum()
        assert cont == cont2
    



    
    
