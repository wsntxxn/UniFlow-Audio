# nohup bash ./bash_scripts_ali/infer.sh > logs/infer.log 2>&1 &
export HF_HOME="/nas-wulanchabu/jiahao.mei/hf_home"
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs_ali/accelerate/nvidia/1gpu.yaml inference.py \
    --config-path configs_ali \
    --config-name inference \
    data@data_dict=sr \
    +exp_dir=experiments/small/small+layer+DoubleContent+2gpus \
    +use_best=false \
    # infer_args.guidance_scale=0.0 
    # +ckpt_dir=experiments/tta/epoch_81



    # exp_dir=experiments/tts/ljspeech/VaribleLengthAudioDiffusion_24Khz_lr1.5e-4_80epoch \

    # epoch_length=1000 \
    # epochs=100 \
    # train_dataloader.batch_size=12 \
    # val_dataloader.batch_size=12 \
    # model=diffusion_singing \
    # loss@loss_fn=weighted_sum \
    # exp_dir=experiments/tts/ljspeech \
    # optimizer.lr=5e-5 \
    # loss_fn.weights.local_duration_loss=0.0 \
    # trainer.wandb_config.project=singing_popcs \
    # trainer.wandb_config.name=vae_wave_backbone_audioudit_layers_24_dim_1024_loss_diff
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/opencpop/checkpoints/epoch_51
    # ~trainer.wandb_config \