# Searchless Chess with Categorical Gaussian Distribution Value Prediction

Pytorch implementation of the papers [Grandmaster-Level Chess Without Search](https://arxiv.org/pdf/2402.04494) and [Stop Regressing: Training Value Functions via
Classification for Scalable Deep RL](https://arxiv.org/pdf/2403.03950). A chess model is trained to predict the action value of a given board state and action by converting the value target to a Gaussian distribution and using categorical cross-entropy loss.

## Setup:

Clone the repository:

```bash
git clone https://github.com/AlxSp/gauss-searchless-chess
cd gauss-searchless-chess
```

### Requirements:

- Python 3.10

Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

### Download Data:

To download the data, run the following command:

```bash
cd data
./download.sh
```

After the download is complete, the data directory should have the following structure:
```
data/
├── train/
|  ├── action_value-00000-of-02148_data.bag
|  ...
|  └── action_value-xxxxx-of-xxxxx_data.bag
├── test/
|  ├── action_value-00000-of-02148_data.bag
|  ...
|  └── action_value-xxxxx-of-xxxxx_data.bag
└── download.sh
```

## Training:

To train the model, run the following command:

```bash
python train.py
```

### Training Configuration Variables
This section describes the configuration variables used in the training script which are found in at the beginning of the `train.py` script.

#### Initialization and Resumption Settings
- `init_from` (str): Determines whether to start training from scratch or resume from a saved model.

  - **scratch**: Start training from scratch.
  - **resume**: Resume training from a saved model.

- `resume_src` (str): Determines the checkpoint to resume training from when init_from is set to '**resume**'.

  - **train**: Resume from the last training checkpoint.
  - **eval**: Resume from the best evaluation checkpoint.
  
#### Model Configuration
- `additional_token_registers` (int): Additional tokens that will be added to the model input.

#### Training Parameters
- `train_save_interval` (int): Interval (in batch iterations) at which the training checkpoint is saved.
- `eval_interval` (int): Interval (in batch iterations) at which the model is evaluated during training.
- `num_epochs` (int): Number of epochs for training.
- `batch_size` (int): Number of samples per batch.
  
#### Learning Rate and Optimization
- `bipe_scale` (float): Batch iterations per epoch scale. Can be used to adjust the learning rate schedule.
- `warmup_steps_ratio` (float): Ratio of warmup iterations to the first epoch.
- `start_lr` (float): Initial learning rate during warmup.
- `max_lr` (float): Maximum learning rate at the end of warmup.
- `final_lr` (float): Final learning rate of the cosine annealing schedule.
- `grad_clip` (float): Gradient clipping value to prevent exploding gradients.

#### Miscellaneous
- `random_seed` (int): Random seed for reproducibility.
- `dataloader_workers` (int): Number of workers for the train and eval dataloaders.
  
### Logging
- `wandb_log` (bool): Set to True to enable logging to Weights and Biases.
- `wandb_project` (str): Name of the Weights and Biases project.
- `wandb_run_name` (str): Name of the Weights and Biases run.

### Checkpoint Directory

The checkpoint directory contains the following files:

```
out/
├── train/
|  ├── model.pt
|  ├── optimizer.pt
|  └── train_state.json
├── eval/
|  ├── model.pt
|  ├── optimizer.pt
|  └── eval_state.json
└── model_config.json
```

#### Directory Structure
- `train/` : most current training checkpoint
- `eval/` : best evaluation checkpoint

#### Files
- `model.pt`: Model checkpoint.
- `optimizer.pt`: Optimizer checkpoint.
- `train_state.json`: Training state variables.
- `eval_state.json`: Evaluation state variables.
- `model_config.json`: Model configuration.

## References

- [Google Deepmind Searchless Chess Implementation](https://github.com/google-deepmind/searchless_chess)

## Citations

```bibtex
@misc{ruoss2024grandmasterlevelchesssearch,
      title={Grandmaster-Level Chess Without Search}, 
      author={Anian Ruoss and Grégoire Delétang and Sourabh Medapati and Jordi Grau-Moya and Li Kevin Wenliang and Elliot Catt and John Reid and Tim Genewein},
      year={2024},
      eprint={2402.04494},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.04494}, 
}
```

```bibtex
@misc{farebrother2024stopregressingtrainingvalue,
      title={Stop Regressing: Training Value Functions via Classification for Scalable Deep RL}, 
      author={Jesse Farebrother and Jordi Orbay and Quan Vuong and Adrien Ali Taïga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
      year={2024},
      eprint={2403.03950},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.03950}, 
}
```