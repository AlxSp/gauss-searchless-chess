#%%
from contextlib import nullcontext
import json
import math
import random
import numpy as np
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from dataset import ActionValueDataset
from schedulers import CosineLearningRateScheduler

import torch
from torch.nn import functional as F

from model import BidirectionalPredictor, PredictorConfig

#%%
init_from = "scratch" # set to "resume" to resume training from a saved model, set to "scratch" to start training from scratch
resume_src = "train"

assert init_from in {"resume", "scratch"}, "init_from must be either 'resume' or 'scratch'"
assert resume_src in {"train", "eval"}, "you must decide where to restart from 'train' or 'eval'"

additional_token_registers = 2 # additional tokens that will be added to the model input
train_save_interval = 20
test_interval = 20
num_epochs = 1
batch_size = 2048

bipe_scale = 1.25 # batch iterations per epoch scale
warmup_steps_ratio = 0.2 # setting it 20% of the first epoch
max_lr = 0.000625 # 0.001
start_lr = 0.0002
final_lr = 1.0e-06
grad_clip = 1.0

random_seed = 42

wandb_log = False #True # set to True to log to wandb
wandb_project = "gauss-searchless-chess"
wandb_run_name = "v1-with-regs"

#%%
# create output directory to store trained model
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

model_config_path = os.path.join(output_dir, "model_config.json")

train_model_dir = os.path.join("out", "train")
eval_model_dir = os.path.join("out", "eval")
os.makedirs(train_model_dir, exist_ok=True)
os.makedirs(eval_model_dir, exist_ok=True)

train_model_path = os.path.join(train_model_dir, "model.pt")
train_optimizer_path = os.path.join(train_model_dir, "optimizer.pt")
train_state_path = os.path.join(train_model_dir, "train_state.json")
eval_model_path = os.path.join(eval_model_dir, "model.pt")
eval_optimizer_path = os.path.join(eval_model_dir, "optimizer.pt")
eval_state_path = os.path.join(eval_model_dir, "eval_state.json")

# create dataset loader
train_data_dir = os.path.join("data", "train")

train_files = [os.path.join(train_data_dir, f) for f in listdir(train_data_dir) if isfile(join(train_data_dir, f)) and f.startswith("action_value")]

train_dataset = ActionValueDataset(train_files, hl_gauss=True, registers= additional_token_registers)

test_data_dir = os.path.join("data", "test")

test_files = [os.path.join(test_data_dir, f) for f in listdir(test_data_dir) if isfile(join(test_data_dir, f)) and f.startswith("action_value")]

test_dataset = ActionValueDataset(test_files, hl_gauss=True, registers= additional_token_registers)

#%%
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

#%%
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

#%%
if init_from == "scratch":
    set_seed(random_seed)
    print("Starting training from scratch")
    iter_num = 0
    model = BidirectionalPredictor(model_config)

    with open(model_config_path, "w") as f:
        json.dump(model_config.__dict__, f, indent=2)
    train_state = {}
elif init_from == "resume":
    if resume_src == "train":
        model_path = train_model_path
        state_path = train_state_path
        optimizer = train_optimizer_path
    elif resume_src == "eval":
        model_path = eval_model_path
        state_path = eval_state_path
        optimizer_path = eval_optimizer_path
    
    print("Resuming training")
    if os.path.exists(model_path):
        train_model_config = PredictorConfig.from_json(model_config_path)
        # load model
        model = BidirectionalPredictor(train_model_config)
        model.load_state_dict(torch.load(model_path))

        with open(state_path, "r") as f:
            train_state = json.load(f)
        iter_num = train_state['iter_num'] + 1
    else:
        raise ValueError("Model file does not exist")

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

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

model.to(device)

save_model = model # create a reference to the non-compiled model, it shares the weights with the compiled model. (Compiled models currently can not be loaded)
if device == "cuda":
    model = torch.compile(model)
else:
    model = torch.compile(model, backend="aot_eager")

# init optimizer
optimizer = torch.optim.AdamW(model.parameters())
if init_from == "resume": # if resuming training, load optimizer state (has to be done after model is moved to device)
    optimizer.load_state_dict(torch.load(optimizer_path))

class ResumableSampler:
    '''
    A continuous sampler, which can be resumed from a given offset.
    '''
    def __init__(self, dataset_len, offset = 0, batch_size = 1):
        self.dataset_len = dataset_len
        self.iter = offset
        self.batch_size = batch_size
        self.batch_iterations_per_epoch = math.ceil(self.dataset_len / self.batch_size)

    def __len__(self):
        return self.batch_iterations_per_epoch
    
    def __iter__(self):
        while True:
            batch_idx = self.iter % self.batch_iterations_per_epoch * self.batch_size
            yield list(range(batch_idx, min(batch_idx + self.batch_size, self.dataset_len)))
            self.iter += 1

train_sampler = ResumableSampler(
    len(train_dataset), 
    offset = iter_num, 
    batch_size = batch_size
)

# create dataloader
# FIXME: setting shuffle to True causes OOM error. Need to create own Sampler which stores indices in file?
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = train_sampler, num_workers=0, pin_memory=True)
train_loader_iter = iter(train_loader)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
#test_loader_iter = iter(test_loader)

max_iter_num = num_epochs * train_sampler.batch_iterations_per_epoch

lr_scheduler = CosineLearningRateScheduler(
    optimizer,
    warmup_steps=int(warmup_steps_ratio * train_sampler.batch_iterations_per_epoch),
    start_lr=start_lr,
    max_lr=max_lr,
    final_lr=final_lr,
    T_max=int(bipe_scale * num_epochs * train_sampler.batch_iterations_per_epoch),
    step=iter_num 
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
        },
        resume = True if init_from == "resume" else False
        )

#%%
train_loss = None
best_eval_loss = train_state.get('best_eval_loss', float("inf"))
p_bar = tqdm(total=max_iter_num, initial=iter_num, desc="Training")
while iter_num < max_iter_num:
    sequence, return_bucket = next(train_loader_iter)
    # for sequence, return_bucket in tqdm(train_loader):
    sequence, return_bucket = sequence.to(device, non_blocking=True), return_bucket.to(device, non_blocking=True)
    #FIXME: maybe split target from sequence
    with type_casting:
        output = model(sequence)
    # currently the computed  "target logits" are taken from the computed output of the action input
    value_logits = output[:, -1, :]
    # we only care about the value logits
    train_loss = F.cross_entropy(value_logits, return_bucket)

    scaler.scale(train_loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    _new_lr = lr_scheduler.step()

    if (iter_num + 1) % train_save_interval == 0:
        torch.save(save_model.state_dict(), train_model_path)
        torch.save(optimizer.state_dict(), train_optimizer_path)

        train_state = {
            'iter_num' : iter_num,
        }

        with open(train_state_path, "w") as f:
            json.dump(train_state, f, indent=2)
    
    mean_eval_loss = 0
    if (iter_num + 1) % test_interval == 0:
        with torch.no_grad():
            for sequence, return_bucket in test_loader:
                sequence, return_bucket = sequence.to(device, non_blocking=True), return_bucket.to(device, non_blocking=True)
                #FIXME: maybe split target from sequence
                with type_casting:
                    output = model(sequence)
                # currently the computed  "target logits" are taken from the computed output of the action input
                value_logits = output[:, -1, :]
                # we only care about the value logits
                mean_eval_loss += F.cross_entropy(value_logits, return_bucket)
            mean_eval_loss /= len(test_loader)

            if mean_eval_loss < best_eval_loss:
                torch.save(save_model.state_dict(), eval_model_path)
                torch.save(optimizer.state_dict(), eval_optimizer_path)
                train_state = {
                    'iter_num' : iter_num,
                    'best_eval_loss' : mean_eval_loss
                }
                with open(eval_state_path, "w") as f:
                    json.dump(train_state, f, indent=2)
                
    
        if wandb_log:
            wandb.log({
                "eval/loss": mean_eval_loss
            }
            , step=iter_num*batch_size)

    if wandb_log:
        wandb.log({
            'train/loss': train_loss.item(),
            'lr': _new_lr
        }
        , step=iter_num * batch_size)
    

    iter_num += 1
    p_bar.update(1)
# %%
