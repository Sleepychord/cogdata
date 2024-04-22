from cogdata.streaming import instantiate_from_yaml, MetaDistributedWebDataset, to_state, mixed_collate
from transformers import AutoTokenizer
from copy import deepcopy
from torch.utils.data import DataLoader

def test_reload_multiworker(conf_path='tests/streaming/merge_testcase.yaml'):
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
    it = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=mixed_collate).__iter__()
    it_2 = DataLoader(dataset2, batch_size=2, num_workers=2, collate_fn=mixed_collate).__iter__()
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
    dataset2.set_dataloader_states(saved_dataloader_states2)
    it2 = DataLoader(dataset2, batch_size=1, num_workers=2, collate_fn=mixed_collate).__iter__()
    # actuall some different due to acc_weights, select a circulating point
    value_dict = {}
    # breakpoint()
    for i in range(50):
        result = next(it_2)
        state = to_state(result)
        for k, v in state.items():
            if k not in value_dict:
                value_dict[k] = v

    for i in range(30):
        result = next(it2)
        state = to_state(result)
        for k, v in state.items():
            if k in value_dict:
                if value_dict[k] is None:
                    continue
                assert value_dict[k] == v
                value_dict[k] = None
            else:
                print('not find:')
                print(k, v)