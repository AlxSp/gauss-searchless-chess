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

#%%
learning_rate = 1e-4

batch_size = 2048

grad_clip = 1.0

wandb_log = False # set to True to log to wandb
wandb_project = "gauss-searchless-chess"
wandb_run_name = "v1"

# create dataset loader
train_dir = os.path.join("data", "train")

train_files = [os.path.join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f)) and f.startswith("action_value")]

train_dataset = ActionValueDataset(train_files, hl_gauss=True)

#%%
# create dataloader
# FIXME: setting shuffle to True causes OOM error. Need to create own Sampler which stores indices in file?
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
model_config = PredictorConfig(
    n_layer = 8,
    n_embd = 256,
    n_head = 8,
    vocab_size = NUM_ACTIONS,
    output_size = train_dataset.num_return_buckets,
    block_size = train_dataset.sample_sequence_length,
    rotary_n_embd = 32,
    dropout = 0.0,
    bias = True,
)

model = BidirectionalPredictor(model_config)

model.to(device)
if device == "cuda":
    model = torch.compile(model)
else:
    model = torch.compile(model, backend="aot_eager")

#%%
# init optimizer
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#%%
if wandb_log:
    import wandb
    wandb.init(
        project=wandb_project, 
        name=wandb_run_name, 
        config=
            {
            'train_config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'grad_clip': grad_clip,
            },
            'model_config':  model_config.__dict__ | {"n_params": model.get_num_params()},
        }
        )

#%%
iter_num = 0
loss = None

for sequence, return_bucket in tqdm(train_loader):
    sequence, return_bucket = sequence.to(device, non_blocking=True), return_bucket.to(device, non_blocking=True)
    #FIXME: maybe split target from sequence
    with type_casting:
        output = model(sequence)
    # currently the computed  "target logits" are taken from the computed output of the action input
    value_logits = output[:, -2, :]
    # we only care about the value logits
    loss = F.cross_entropy(value_logits, return_bucket)

    scaler.scale(loss).backward()

    if wandb_log:
        wandb.log({
            'train/loss': loss.item(),
        }
        , step=iter_num * batch_size)

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    iter_num += 1