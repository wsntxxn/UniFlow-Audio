# Evaluation Guide

This document explains how to run evaluation for various tasks.

## 1. Setup

It is recommended to create a separate environment for evaluation, as described in [requirements_eval.txt](../requirements_eval.txt).

```bash
conda create -n uniflow_audio_eval python=3.10
conda activate uniflow_audio_eval
```

Then, install the dependencies:

```bash
pip install torch torchaudio torchvision
pip install nemo_toolkit['all']
pip install -r requirements_eval.txt
```

**Troubleshooting:** If you encounter `ImportError: cannot import name 'functional_tensor'` when running V2A evaluation, a simple but not elegant way is to modify `pytorchvideo/transforms/augmentations.py` line 9:

```python
# Change from:
import torchvision.transforms.functional_tensor as F_t
# To:
import torchvision.transforms.functional as F_t
```

## 2. Prepare Data

Prepare the test dataset for each task.
Most datasets follow the same format in the training process.
Some tasks or datasets may need specific formats.
You can freely adjust the format requirement by modifying the evaluation code.

### T2A & T2M

Due to the requirement of [audioldm_eval](https://github.com/haoheliu/audioldm_eval), the test files of a dataset should be placed in the same folder.
Then the corresponding `audio.jsonl` needs to be generated.

### V2A

Evaluation of V2A involves [audioldm_eval](https://github.com/haoheliu/audioldm_eval) and [GMELab](https://github.com/ilpoviertola/GMELab), so the original VisualSound video data need to be clipped and resampled:

**1. Resample audio to 16kHz**
```bash
export PYTHONPATH=.
python data_preprocess/vggsound/extract_wav.py \
    --audio_jsonl data/visual_sound/test_videos.jsonl \
    --output_dir data/visual_sound/test_wav_16000Hz_0s_to_10.0s \
    --output_jsonl data/visual_sound/test_wav_16000Hz_0s_to_10.0s.jsonl \
    --vggsound_label_path /path/to/label/csv
```

> **Note:** `vggsound_label_path` is the [original VGGSound annotation file](https://raw.githubusercontent.com/hche11/VGGSound/refs/heads/master/data/vggsound.csv).

**2. Reencode video to 25fps**
```bash
python data_preprocess/vggsound/reencode_video.py \
    -i data/visual_sound/test_videos.jsonl \
    --video_fps 25 \
    --audio_fps 16000 \
    -o data/visual_sound/test_video_fps_25_sr_16000 \
    -j data/visual_sound/test_video_fps_25_sr_16000.jsonl
```

**3. Extract ImageBind visual embeddings**
```bash
python data_preprocess/vggsound/imagebind_vision_embeds.py \
    -j data/visual_sound/test_videos.jsonl \
    -o data/visual_sound/test_ib_visual_embed.h5 \
```

## 3. Run Evaluation

[eval_all.sh](../bash_scripts/eval_all.sh) does the evaluation for all tasks:
```bash
bash bash_scripts/eval_all.sh \
    --infer_dir /path/to/inference/directory \
    --exp_name "your_exp_name"
```
where environment variables and data paths need to be modified correspondingly.

Below are evaluation guides for single tasks.
First set environment variables:
```bash
export PYTHONPATH=.:evaluation/GMELab
export CLAP_MODEL_PATH="/path/to/laion_clap/630k-audioset-best.pt"
```

### TTS

```bash
python evaluation/tts.py \
  --audio_dir /path/to/tts_inference \
  --xp_name "your_exp_name" \
  --ref_transcript_path data/libritts/test/ref_transcription.json
  --ref_audio_path data/libritts/test/ref_audio.json \
  --output_path /path/to/tts_result.txt
```

where `ref_audio_path` is a simple JSON file like:

```JSON
{
  "5290_26685_000040_000002": "/path/to/LibriTTS/train-clean-360/5290/26685/5290_26685_000040_000002.wav",
  "5290_26685_000069_000001": "/path/to/LibriTTS/train-clean-360/5290/26685/5290_26685_000069_000001.wav",
  "5290_26685_000052_000001": "/path/to/LibriTTS/train-clean-360/5290/26685/5290_26685_000052_000001.wav",
  // ...
}
```

### SVS

```bash
python evaluation/svs.py \
    --ref_audio_jsonl data/m4singer/test/audio.jsonl \
    --gen_audio_dir /path/to/svs_inference \
    --output_file /path/to/svs_result.txt
```

### T2A & T2M

```bash
python evaluation/t2a.py \
    --ref_audio_jsonl /path/to/t2a_or_t2m/audio.jsonl \
    -rc /path/to/t2a_or_t2m/caption.jsonl \
    -gd /path/to/t2a_or_t2m_inference \
    -o /path/to/t2a_or_t2m_results.txt
```

### SE

```bash
python evaluation/se.py \
    --ref_dir /path/to/clean/speech/dir \
    --gen_dir /path/to/se_inference \
    --uuid_jsonl /path/to/uuid.jsonl \
    --output_file /path/to/se_results.txt
```
where `uuid_jsonl` follows this format (due to historical reasons there are such inconsistencies somewhere):

```JSON
{"UUID": "5d247b58-bd56-4d53-9a21-b5e24365686a", "WavPath": "clean_testset_wav/p232_002.wav"}
{"UUID": "33ecf74d-ba9e-4cc4-8ae5-51f53601aa3a", "WavPath": "clean_testset_wav/p232_017.wav"}
{"UUID": "1051c94e-bbcc-4b58-888d-53457ae9376b", "WavPath": "clean_testset_wav/p232_021.wav"}
```

### SR

Evaluation of super resolution is integrated into a single bash script:

```bash
bash bash_scripts/eval_audio_sr.sh /path/to/sr_inference
```

The reference base directory argument `-rb` should be set properly. 

### V2A

```bash
python evaluation/v2a.py \
    -ra data/visual_sound/test_audio_16000Hz_0s_to_10.0s.jsonl \
    -ibv data/visual_sound/ib_visual_embed.h5 \
    -syncv data/visual_sound/test_videos_fps_25_sr_16000.jsonl \
    -gd /path/to/v2a_inference \
    -o /path/to/v2a_results.txt
```

---

Evaluation results will be saved to the output arguments specified by `-o`/`--output_file`/`--output_path`.


## 4. Notes

The evaluation code and toolkit integrate evaluation tools from all tasks, so the code may not be fully optimized and some unexpected issues might occur during environment setup. You may need to inspect the evaluation code and slightly modify it.