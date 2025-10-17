# :sound: UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities

[![arXiv](https://img.shields.io/badge/arXiv-2509.24391-brightgreen.svg?logo=Arxiv&style=flat-square)](https://arxiv.org/abs/2509.24391)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://wsntxxn.github.io/uniflow_audio) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wsntxxn/UniFlow-Audio) [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow?style=flat-square)](https://huggingface.co/wsntxxn/UniFlow-Audio-large)

This repository is the official implementation of the paper "[UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities](https://arxiv.org/abs/2509.24391)".
We provide a lightweight training framework, built on [Accelerate](https://huggingface.co/docs/accelerate/index).
It can be customized easily with all code exposed in [trainer.py](./train.py).

## 💡 Inference using Pre-trained Models

### Dependency Installation

First, please install dependencies required for training and inference.

```bash
conda create -n uniflow-audio python=3.10
```
If you want to perform text-to-speech (TTS) synthesis inference, you also need to install `montreal-forced-aligner`, so executing the following command instead:
```bash
conda create -n uniflow-audio -c conda-forge python=3.10 montreal-forced-aligner
```
Then install python dependencies:
```bash
conda activate uniflow-audio
pip install -r requirements.txt
```

**Optional Dependencies for TTS Inference**

To extract speaker embedding for TTS inference, you need to install `wespeaker`: 

```bash
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

**Optional Dependencies for V2A Inference**

To perform video-to-audio (V2A) generation inference, please install the following additional libraries:

```bash
pip install moviepy av torchvision
```

### Running Inference

Please refer to [INFERENCE_CLI.md](./docs/INFERENCE_CLI.md) for inference CLI examples.

## :hammer_and_wrench: Training

### Data Format

For each generation dataset, the input content information should be organized in a `content.jsonl`.
Each line in `content.jsonl` is like:
```JSON
{"audio_id": "xxx", "caption": "xxx"}
```

The target audio files should be organized in an `audio.jsonl`, with similar formats:
```JSON
{"audio_id": "xxx", "audio": "/path/to/audio/file"}
```

Then, for each task type, implement a class by inheriting `AudioGenerationDataset` in `data_module/dataset.py`: the content loading method is defined here.

For datasets used in the paper, our pre-processing scripts are in [data_preprocess](./data_preprocess).
You may use them as reference to process your own data.

### Configurations

We use `hydra` + `omegaconf` to organize training configurations.
* `hydra` organizes the configuration into separate modules by [defaults list](https://hydra.cc/docs/advanced/defaults_list), and supports command line overrides. See docs and examples in `configs`.
* `omegaconf` supports [custom resolvers](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#id9) with [native variable interpolations](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation), so fields in YAML can be set more dynamically.
See above docs for more details.

#### Hydra Override Examples
Here are some hydra override examples:

##### Example 1
```bash
python inference.py +data_dict.audiocaps.test.max_samples=100
```
It sets the maximum number of samples for the `test` split of [audiocaps](./configs/data/datasets/audiocaps.yaml) dataset to 100.

##### Example 2
```bash
accelerate launch train.py \
  model/backbone=input_fusion_dit
```
It uses `input_fusion_dit` instead of the original `layer_fusion_dit`.
This is an example of overriding a config group that is not at the top level.

### Customize Training

Like pytorch-lightning, this framework makes a little abstraction on the native PyTorch-based training loop, making training on new models, datasets and loss functions easier.
The most efforts lie in implementing these components and write YAML configs correspondingly:
1. Implement datasets, models, loss functions...: This is the same as normal PyTorch-based training pipeline.
2. Implement custom trainer:  Similar to `LightningModule` in pytorch-lightning, we define a bunch of hooks in the training loop. To customize the training process, minimally we just need to define the behavior of `training_step` and `validation_step`. We can also customize other hooks, such as `on_train_start` and `on_validation_start`. [audio_generation_trainer.py](audio_generation_trainer.py) gives an example.
3. Write YAML files: YAML configs need to be configured to use the dataset, model, ..., and trainer defined above. Among them, "train_dataloader", "val_dataloader", "optimizer", "lr_scheduler" and "loss_fn" must be specified.

The YAML format is hydra-style, for example:
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
Training is launched by `accelerate` command line tool:
```bash
accelerate launch train.py
# or
accelerate launch train.py --config-path path/to/config/dir --config-name conf 
```
This will use `path/to/config/dir/conf.yaml` as the configuration entrypoint, and `${HF_HOME}/accelerate/default_config.yaml` for accelerate configuration.

Command line overrides are stil supported:
```bash
accelerate launch --config_file configs/accelerate/8gpus.yaml train.py \
    warmup_params.warmup_steps=500 \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    epochs=100
```

### Inference

After training, experiment logging files, checkpoints, and other artifacts are saved in `${exp_dir}` defined in `configs/train.yaml`.
We still use `accelerate` to do inference:
```bash
exp_dir="/path/to/exp_dir"
ckpt_dir="/path/to/exp_dir/checkpoints/epoch_xxx"
accelerate launch \
  inference.py \
  data@data_dict=t2a_audiocaps \
  exp_dir=${exp_dir} \
  ckpt_dir_or_file=${ckpt_dir}
```
This will infer on AudioCaps test set with the default configurations in `configs/inference.yaml`.

## :bar_chart: Evaluation

For evaluation, please refer to [EVALUATION.md](./docs/EVALUATION.md). 

## :memo: TODO
- [x] Add inference script for pre-trained models.
- [x] Add README about evaluation guidance.
- [x] Add interactive inference interface link. 

## :book: Citation

If you found the paper or the codebase useful, please consider citing
```bibtex
@article{xu2025uniflow,
  title={UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities},
  author={Xu, Xuenan and Mei, Jiahao and Zheng, Zihao and Tao, Ye and Xie, Zeyu and Zhang, Yaoyun and Liu, Haohe and Wu, Yuning and Yan, Ming and Wu, Wen and Zhang, Chao and Wu, Mengyue},
  author={Zheng, Zihao and Xie, Zeyu and Xu, Xuenan and Wu, Wen and Zhang, Chao and Wu, Mengyue},
  journal={arXiv preprint arXiv:2509.24391},
  year={2025}
}
```

## :sparkles: Acknowledgements

We would like to express our gratitude to the following projects and their contributors, from which we have borrowed code or drawn inspiration:

- **[EzAudio](https://github.com/haidog-yaqub/EzAudio)**
- **[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)**
- **[Tango](https://github.com/declare-lab/tango)**

We appreciate the open-source community for making these valuable resources available.
