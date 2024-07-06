# Searchless Chess

Pytorch implementation of:

[Grandmaster-Level Chess Without Search](https://arxiv.org/pdf/2402.04494) and [Stop Regressing: Training Value Functions via
Classification for Scalable Deep RL](https://arxiv.org/pdf/2403.03950)

## Data:

The data directory is expected to contain the following structure:

```
data
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

## Todo:

- [X] logging
- [X] check if tokenizer is correctly implemented
- [X] save the model and optimizer state
- [X] allow resuming from a saved model
- [ ] test set evaluation
- [X] cosine learning rate schedule
- [ ] weight decay?