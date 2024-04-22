import torch

class FakeDataset(torch.utils.data.Dataset):
    '''Fake dataset for testing purposes, each samples is 000..00 with 256 length'''
    def __init__(self, length=256):
        self.length = length
    def __len__(self):
        return 10000000
    def __getitem__(self, idx):
        return {'zeros':'0'*256}

    