# CUDA_VISIBLE_DEVICES=1 bash bash_scripts/infer_batch.sh

# ------------------------------
export HF_HOME='/hpc_stor03/sjtu_home/xuenan.xu/hf_cache'
ckpt_dir=/hpc_stor03/sjtu_home/jiahao.mei/code/x_to_audio_generation/experiments/base_dummy_layer_fusion/ckpt_step_400000 #specify ckpt dir here, ckpt_path:ckpt_dir / "model.safetensors"
GPUS=1gpu #[1gpu,2gpus,4gpus,8gpus]
wav_dir_root='./xps'
guidance_scale=3.0
num_steps=20
dataset_name=tta #define in ./configs/data/default.yaml
exp_config_path=/hpc_stor03/sjtu_home/jiahao.mei/code/x_to_audio_generation/experiments/base_dummy_layer_fusion/"config.yaml"
infer_dir="${dataset_name}_gs${guidance_scale}_steps${num_steps}" # output dir,generated samples save in wav_dir_root/infer_dir
# ------------------------------

accelerate launch --config-file configs/accelerate/nvidia/${GPUS}.yaml inference.py \
    infer_args.guidance_scale=${guidance_scale} \
    infer_args.num_steps=${num_steps} \
    wav_dir=${infer_dir} \
    data@data_dict=${dataset_name} \
    ckpt_dir=${ckpt_dir} \
    exp_dir=${exp_dir} \
    +wav_dir_root=${wav_dir_root} \
    +exp_config_path=${exp_config_path}