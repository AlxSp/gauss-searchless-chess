
#%%
import os
import torch
from torch.utils.data import Dataset, DataLoader
from bagz import BagReader, BagDataSource
#%%
class ChessDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

        self.lengths = []
        for file_path in self.file_paths:
            self.lengths.append(len(BagReader(file_path)))

        self.length = sum(self.lengths)

    def _get_record_index(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return i, idx
            idx -= length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #TODO: add out of bounds check
        file_idx, record_idx = self._get_record_index(idx)
        return BagReader(self.file_paths[file_idx])[record_idx]
    
#%%
class ChessDataCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        pass
    
#%%
from os import listdir
from os.path import isfile, join

train_dir = "/ubuntu_data/searchless_chess/data/train"

train_files = [os.path.join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))]

#%%
b = BagReader(train_files[0])

len(b)


#%%
ds = ChessDataset(train_files)
len(ds)

#%%
ds._get_record_index(11177425)

#%%
len(ds)


ds[1637485632]