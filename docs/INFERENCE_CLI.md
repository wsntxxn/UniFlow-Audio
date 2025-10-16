# Inference Guide

## Single-instance Inference

All single-instance inference can be performed by calling [inference_cli.py](../inference_cli.py).
Below are examples for different tasks:

### Text-To-Speech (TTS)
```bash
python inference_cli.py tts \
    --transcript "Hello this is a special sentence with zyloph" \
    --ref_speaker_speech ./data/egs/tts_speaker_ref.wav \
    --model_name "UniFlow-Audio-large" \
    --output_path speech.wav
```

### Singing-Voice-Synthesis (SVS)
```bash
python inference_cli.py svs \
    --singer Alto-2 \
    --music_score "AP你要相信AP相信我们会像童话故事里AP<sep>rest | G#3 | A#3 C4 | D#4 | D#4 F4 | rest | E4 F4 | F4 | D#4 A#3 | A#3 | A#3 | C#4 | B3 C4 | C#4 | B3 C4 | A#3 | G#3 | rest<sep>0.14 | 0.47 | 0.1905 0.1895 | 0.41 | 0.3005 0.3895 | 0.21 | 0.2391 0.1809 | 0.32 | 0.4105 0.2095 | 0.35 | 0.43 | 0.45 | 0.2309 0.2291 | 0.48 | 0.225 0.195 | 0.29 | 0.71 | 0.14" \
    --model_name "UniFlow-Audio-large" \
    --output_path singing.wav
```
Here available singers are from [M4Singer](https://m4singer.github.io/), which can be found in [spk_set.json](https://huggingface.co/wsntxxn/UniFlow-Audio-large/blob/main/svs/spk_set.json).
Music scores are represented as sequences separated by `<sep>`: lyric, note name (e.g., C4, F#3, rest), and note duration.

### Text-to-Audio (T2A) / Text-to-Music (T2M)
```bash
python inference_cli.py t2a \
    --caption "a man is speaking then a dog barks" \
    --model_name "UniFlow-Audio-large" \
    --output_path audio.wav
python inference_cli.py t2m \
    --caption "pop music with a male singing rap" \
    --model_name "UniFlow-Audio-large" \
    --output_path music.wav
```

### Speech Enhancement (SE) / Audio Super-Resolution (SR)
```bash
python inference_cli.py se \
    --noisy_speech ./data/egs/se_noisy_sample.wav \
    --model_name "UniFlow-Audio-large" \
    --output_path clean_speech.wav

python inference_cli.py sr \
    --low_sr_audio ./data/egs/sr_low_sr_sample.wav \
    --model_name "UniFlow-Audio-large" \
    --output_path 24k_sr_audio.wav
```

### Video-to-Audio (V2A)
```bash
python inference_cli.py v2a \
    --video ./data/egs/v2a_video_sample.mp4 \
    --model_name "UniFlow-Audio-large" \
    --output_path video.mp4
```


## Batch inference

For batch inference, please use [inference.py](../inference.py) as [the practice in training](../README.md#inference).
You just need to specify a dummy experiment directory and the checkpoint path.
Assume the model on HuggingFace is downloaded to `${local_dir}`:
```bash
accelerate launch \
    inference.py \
    data@data_dict=tta_audiocaps \
    exp_dir=${local_dir} \
    ckpt_dir_or_file=${local_dir}/model.safetensors \
    wav_dir_root=./ \
    wav_dir=batch_inference
``` 
Results will be saved to `./batch_inference`.
The corresponding inference config is in [inference.yaml](../configs/inference.yaml) and the dataset used for inference is in [t2a_audiocaps.yaml](../configs/data/t2a_audiocaps.yaml).