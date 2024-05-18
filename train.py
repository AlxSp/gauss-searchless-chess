#%%
from contextlib import nullcontext
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

train_files = [os.path.join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f)) and f.startswith("action_value")]

ds = ActionValueDataset(train_files)

#%%
# create dataloader
# FIXME: setting shuffle to True causes OOM error. Need to create own Sampler which stores indices in file?
train_loader = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)

#%%
# create model

# ~9M model
if torch.backends.mps.is_available():
    device_type = 'mps'
elif torch.cuda.is_available():
    device_type = 'cuda'
else:
    device_type = 'cpu'

device = torch.device(device_type)
print(f"Device: {device_type}")

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
type_casting = nullcontext() if device_type in {'cpu', 'mps'} else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
if device == "cuda":
    model = torch.compile(model)
else:
    model = torch.compile(model, backend="aot_eager")

#%%
# init optimizer
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
grad_clip = 1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

#%%
count = 0
loss = None

for sequence, loss_mask in tqdm(train_loader):
    sequence, loss_mask = sequence.to(device, non_blocking=True), loss_mask.to(device, non_blocking=True)
    #FIXME: maybe split target from sequence
    with type_casting:
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
