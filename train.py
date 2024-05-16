#%%
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from dataset import ActionValueDataset, NUM_ACTIONS

import torch
from torch.nn import functional as F

from model import BidirectionalPredictor, PredictorConfig
#%%
# create dataset loader
train_dir = "/ubuntu_data/searchless_chess/data/train"

train_files = [os.path.join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))]

ds = ActionValueDataset(train_files)

#%%
# create dataloader
# FIXME: setting shuffle to True causes OOM error. Need to create own Sampler which stores indices in file?
train_loader = torch.utils.data.DataLoader(ds, batch_size=1536, shuffle=False)

#%%
# create model

# ~9M model

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'



#output_size = num_return_buckets
config = PredictorConfig(
    n_layer = 8,
    n_embd = 256,
    n_head = 8,
    vocab_size = NUM_ACTIONS,
    output_size = ds.num_return_buckets,
    block_size = ds.sample_sequence_length,
    rotary_n_embd = 32,
    dropout = 0.0,
    bias = True,
)

model = BidirectionalPredictor(config)

model.to(device)

#%%
# init optimizer
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
grad_clip = 1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

#%%
count = 0
loss = None

for sequence, loss_mask in tqdm(train_loader):
    sequence, loss_mask = sequence.to(device), loss_mask.to(device)
    #FIXME: maybe split target from sequence
    output = model(sequence, attn_mask=loss_mask.unsqueeze(1).unsqueeze(1).bool())
    # currently the computed  "target logits" are taken from the computed output of the action input
    value_logits = output[:, -2, :]
    # we only care about the value logits
    loss = F.cross_entropy(value_logits, sequence[:, -1])

    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
# %%
