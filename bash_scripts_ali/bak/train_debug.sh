CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml train.py \
    epochs=400 \
    trainer.wandb_config.name=opencpop_waveform_vae_udit6layer_diffusion \
    exp_dir=experiments/waveform_vae_diffsinger_diffusion/opencpop/ \
    model=diffusion_singing \
    ~trainer.wandb_config