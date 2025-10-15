# :sound: UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities

[![arXiv](https://img.shields.io/badge/arXiv-2509.24391-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.24391)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://wsntxxn.github.io/uniflow_audio) <!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation) -->

This repository is the official implementation of the paper "[UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities](https://arxiv.org/abs/2509.24391)".
We provide a lightweight training framework, built on [Accelerate](https://huggingface.co/docs/accelerate/index).
It can be customized easily with all code exposed in [trainer.py](./train.py).

## 💡 Inference using Pre-trained Models

### Dependency Installation

First, please install the training dependencies required for the project.

```
conda create -n uniflow-aduio python=3.10
pip install -r requirements.txt
```



**Additional Dependencies for TTS Inference**

To perform Text-to-Speech (TTS) inference, you need to install the following extra dependencies. Note: To avoid potential compatibility issues, please be sure to install `kaldi`, `pynini`, and `montreal-forced-aligner` using Conda.

```bash
conda install -c conda-forge kaldi pynini montreal-forced-aligner
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

**Additional Dependencies for V2A Inference**

To perform Video-to-Audio (V2A) inference, please install the following libraries:

```bash
pip install moviepy av torchvision
```

### Models and Configuration

Before running inference, please ensure you have the following files ready:

*   **Model Checkpoint**: Use the `model.safetensors` we provide and specify its path in the `ckpt_dir` argument of the inference script.
*   **VAE Checkpoint**: Use the `vae.ckpt` we provide and ensure its path is correctly specified in the configuration file.
*   **Configuration File**: Use our static configuration file located at `configs/config.yaml` for inference. This config must be used together with the corresponding model checkpoint. You should specify the path to the VAE checkpoint in the `model.autoencoder.pretrained_ckpt` field within this file.

We design 10 instructions for each task, which you can find in `data/instructions/task_instruction.csv`. These instructions have been encoded using the T5 model and are stored as embeddings in `data/instructions/t5_embeddings.h5` for direct use by the model during inference.

### Running Inference

We support both batch and single-instance inference modes. 

*   **Batch Inference**: To process multiple data samples, use the `bash_scripts/infer_batch.sh` script.
*   **Single Inference**: To process a single data sample, use the `bash_scripts/infer_single.sh` script.

Different inference tasks require different types of input data, as detailed below:

*   **Text-to-Audio (T2A) / Text-to-Music (T2M)**:
    *   `caption`: Provide a plain text description.
*   **Speech Enhancement (SE) / Speech Super-Resolution (SR)**:
    *   `audio_path`: Provide the file path to the audio to be processed.
*   **Video-to-Audio (V2A)**:
    *   `video_path`: Provide the video file path. The video content will be encoded using the CLIP model.
*   **Singing Voice Synthesis (SVS)**:
    *   `speaker_name`: Specify the speaker's name.
    *   `phoneme_sequence`: Provide the phoneme sequence of the lyrics.
    *   `note`: Provide the note sequence.
    *   `note_duration`: Provide the duration of each note.
*   **Text-to-Speech (TTS)**:
    *   `transcript`: Provide the text to be synthesized. This text will be automatically converted to a phoneme sequence using the [Montreal Forced Aligner](https://mfa-models.readthedocs.io/en/latest/index.html).
    *   `ref_speaker_audio`: Provide the reference speaker's audio. Acoustic features (X-vector) will be extracted from this audio using [WeSpeaker](https://github.com/wenet-e2e/wespeaker).

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
  data@data_dict=tta_audiocaps \
  exp_dir=${exp_dir} \
  ckpt_dir=${ckpt_dir}
```
This will infer on AudioCaps test set with the default configurations in `configs/inference.yaml`.

## :bar_chart: Evaluation

For evaluation, please refer to [EVALUATION.md](./docs/EVALUATION.md). 

## :memo: TODO
- [x] Add inference script for pre-trained models.
- [x] Add README about evaluation guidance.

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
