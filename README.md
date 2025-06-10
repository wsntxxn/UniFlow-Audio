# Generation Audio Generation with Instructions


## Data Format




## Config-related Features

We use `hydra` + `omegaconf` to organize the training configuration.
`hydra` organizes the configuration into separate modules, and supports command line overrides.
`omegaconf` supports custom resolvers, so fields in YAML can be set more dynamically.

### Resolvers and Variable Interpolation by omegaconf

* [Variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) supported by omegaconf is used to reuse configuration values in YAML
* Custom resolvers must be registered in [register_omegaconf_resolvers()](utils/config.py#L33)

### Hydra Defaults List
The YAML config uses the feature of [hydra defaults list](https://hydra.cc/docs/advanced/defaults_list/). For example, [this line](configs/default.yaml#L4):
```YAML
defaults:
- data@train_dataloader.dataset.datasets: train
```
indicates that config in `${config_path}/data/train.yaml` (where `${config_path}` is `configs` by default) will be loaded and set as the value of `train_dataloader.dataset.datasets`.

Generally, `a@b: c` means `${config_path}/a/c.yaml` will be loaded and set as the value of `b`.

### Hydra Override Examples

#### Example 1
```bash
python inference.py +data_dict.vggsound_clip.test.max_samples=100
```
When we use `vggsound_clip` data, we need to directly override the most inner field instead of the pointer (pointed by the default list).

#### Example 2
```bash
accelerate launch train.py \
  model/content_adapter=prefix_adapter
```
This is an example of overriding a config group in the config that is not at the top level.

## Configure Training

Like pytorch-lightning, the training framework makes the training on new models, datasets and loss functions easier.
The most efforts lie in implementing these components and write YAML configs correspondingly.

### Implement Dataset, Model, Loss, Lr Scheduler ...
This part is the same as normal PyTorch-based training pipeline.

### Implement Custom Trainer

Similar to `lightning.LightningModule` in pytorch-lightning, we define a bunch of hooks in the training loop.
To customize the training process, minimally we just need to define the behavior of `training_step` and `validation_step`.
We can also add other behavior in many places, e.g., `on_train_start` and `on_validation_start`.

[audio_generation_trainer.py](audio_generation_trainer.py) gives an example.

### Write YAML Files

The most labor-extensive part may be to write YAML configs to use the dataset, model, ..., and trainer defined above.
Among them, "train_dataloader", "val_dataloader", "optimizer", "lr_scheduler" and "loss_fn" must be specified.

The format uses hydra-style, i.e.,
```YAML
object:
  _target_: module.submoule.Class
  param1: value1
  param2: value2
  sub_object:
    _target_: module.submodule.SubClass
    param1: value1
    param2: value2
```
The object will be instantiated recursively. 

### Launch Training
Training is launched through `train.py`:
```bash
accelerate launch train.py
```
To specify the config YAML file:
```bash
accelerate launch train.py --config-path path/to/config/dir --config-name conf 
```
This will use `path/to/config/dir/conf.yaml` as the configuration entrypoint, and `${HF_HOME}/accelerate/default_config.yaml` for accelerate configuration.

It supports convenient override via hydra grammar, e.g.
```bash
accelerate launch --config_file configs/accelerate/8gpus.yaml train.py \
    warmup_params.warmup_steps=500 \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    epochs=100
```

## TODO
- [ ] Implement the fusion of time-independent and time-varying condition
- [ ] Implement evaluation metrics of several tasks