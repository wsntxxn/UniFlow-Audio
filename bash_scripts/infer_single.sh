# CUDA_VISIBLE_DEVICES=1 bash bash_scripts/infer_single.sh

# ============================================================
# 🚀 Basic Configuration: Input and Task Type
# ============================================================


# Task type:
#   t2a: text-to-audio
#   t2m: text-to-music
#   se: speech enhancement
#   sr: speech super-resolution
#   v2a: video-to-audio
#   svs: singing voice synthesis
#   tts: text-to-speech
task=v2a  # Current generation task type 


# The "caption" variable defines the input for generation:
# - For tasks [se, sr, v2a], use an audio or video file path as input.
# - For tasks [t2a, t2m], use plain text as input.
# - For tasks [tts], use  `transcript|reference audio path` as input.
# - For tasks [tts], use  `spkeaer|text|note|note_duration` as input.

caption="data/egs/v2a_video_sample.mp4"  # Input file path (an audio file or video file)
# caption="generate a voilin music"  # Input plain text
# caption="hello world this is a special sentence with zyloph|data/egs/tts_speaker_ref.wav"  # for tts,  Input transcript and reference speaker audio path separated by '|' 
# caption='"Alto-2|<SP> uo zh i d ao m ei l i h uei l ao q v <SP>|[0, 57, 60, 60, 57, 57, 60, 60, 57, 57, 60, 60, 60, 60, 62, 62, 0]|[0.675,0.11,0.41,0.41,0.17,0.17,0.84,0.84,0.33,0.33,0.34,0.34,0.41,0.41,0.97,0.97,0.235]"'  # for svs,caption is `spkeaer|text|note|note_duration`, spaker is defined in data/svs_spk_set.json, phoneme is defined in data/svs_phone_set.json



# The filename index for the generated audio.
# Used to distinguish between multiple generated samples.
audio_id='11'


# Root directory for saving generated audio files.
wav_dir_root='./xps'



# ============================================================
# 🧠 Model and Configuration Paths
# ============================================================

# Path to the Hugging Face model cache directory
export HF_HOME='/hpc_stor03/sjtu_home/xuenan.xu/hf_cache'

# Directory containing model checkpoints.
# This folder should include "model.safetensors".
ckpt_dir=/hpc_stor03/sjtu_home/jiahao.mei/code/x_to_audio_generation/experiments/base_dummy_layer_fusion/ckpt_step_400000

# Path to the pretrain model configuration file (YAML format)
exp_config_path=configs/config.yaml

# ============================================================
# ⚙️ Inference Parameters
# ============================================================

# Number of GPUs to use.
# Possible values: [1gpu, 2gpus, 4gpus, 8gpus]
GPUS=1gpu


# Guidance scale — controls the strength of classifier-free-guidance during generation.
# Higher values increase faithfulness to the input condition
# but may reduce diversity in the output.
guidance_scale=3.0

# Number of inference steps.
num_steps=20

# Output directory name.
# Automatically constructed from task type, guidance scale, and inference steps.
# Example: sr_gs3.0_steps20
infer_dir="${task}_gs${guidance_scale}_steps${num_steps}"
# ============================================================

accelerate launch --config-file configs/accelerate/nvidia/${GPUS}.yaml inference.py \
    infer_args.guidance_scale=${guidance_scale} \
    infer_args.num_steps=${num_steps} \
    wav_dir=${infer_dir} \
    data@data_dict=${task} \
    ckpt_dir=${ckpt_dir} \
    +wav_dir_root=${wav_dir_root} \
    +task=${task} \
    +exp_config_path=${exp_config_path} \
    +single_infer=true \
    +caption="${caption}" \
    +audio_id=${audio_id}


