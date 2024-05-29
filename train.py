#%%
from contextlib import nullcontext
import math
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from dataset import ActionValueDataset, NUM_ACTIONS
from schedulers import CosineLearningRateScheduler

import torch
from torch.nn import functional as F

from model import BidirectionalPredictor, PredictorConfig

#%%
additional_token_registers = 2 # additional tokens that will be added to the model input

batch_size = 2048

grad_clip = 1.0

num_epochs = 1
bipe_scale = 1.25 # batch iterations per epoch scale
warmup_steps_ratio = 0.2 # setting it 20% of the first epoch
max_lr = 0.000625 # 0.001
start_lr = 0.0002
final_lr = 1.0e-06

wandb_log = True # set to True to log to wandb
wandb_project = "gauss-searchless-chess"
wandb_run_name = "v1-with-regs"

# create dataset loader
train_dir = os.path.join("data", "train")

train_files = [os.path.join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f)) and f.startswith("action_value")]

train_dataset = ActionValueDataset(train_files, hl_gauss=True, registers= additional_token_registers)

#%%
# create dataloader
# FIXME: setting shuffle to True causes OOM error. Need to create own Sampler which stores indices in file?
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
batch_iterations_per_epoch = len(train_loader)

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
print(f"Type: {dtype}")
print(f"Using autocast: {device_type not in {'cpu', 'mps'}}")


#output_size = num_return_buckets
model_config = PredictorConfig(
    n_layer = 8,
    n_embd = 256,
    n_head = 8,
    vocab_size = train_dataset.vocab_size,
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

optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)

lr_scheduler = CosineLearningRateScheduler(
    optimizer,
    warmup_steps=int(warmup_steps_ratio*batch_iterations_per_epoch),
    start_lr=start_lr,
    max_lr=max_lr,
    final_lr=final_lr,
    T_max=int(bipe_scale*num_epochs*batch_iterations_per_epoch),
    step=0 
)

#%%
if wandb_log:
    import wandb
    wandb.init(
        project=wandb_project, 
        name=wandb_run_name, 
        config=
            {
            'train_config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'bipe_scale': bipe_scale,
                'warmup_steps_ratio': warmup_steps_ratio,
                'start_lr': start_lr,
                'max_lr': max_lr,
                'final_lr': final_lr,
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
    value_logits = output[:, -1, :]
    # we only care about the value logits
    loss = F.cross_entropy(value_logits, return_bucket)

    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    _new_lr = lr_scheduler.step()

    if wandb_log:
        wandb.log({
            'train/loss': loss.item(),
            'lr': _new_lr
        }
        , step=iter_num * batch_size)
    
    iter_num += 1