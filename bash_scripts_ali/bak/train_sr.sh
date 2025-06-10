CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file /hpc_stor03/sjtu_home/ye.tao/workspace/x_to_audio_generation/configs_ali/accelerate/nvidia/1gpus.yaml /hpc_stor03/sjtu_home/ye.tao/workspace/x_to_audio_generation/train.py \
    epoch_length=10000 \
    epochs=100 \
    data@data_dict=audio_sr \
    train_dataloader.batch_size=2 \
    val_dataloader.batch_size=2 \
    model=diffusion_sr \
    loss@loss_fn=weighted_sum \
    exp_dir=/hpc_stor03/sjtu_home/ye.tao/workspace/experiments/audiosr/24k_new \
    optimizer.lr=5e-5 \
    loss_fn.weights.local_duration_loss=0.1 \
    trainer.wandb_config.project=audio_sr \
    trainer.wandb_config.name=vae_auidosr_new
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/opencpop/checkpoints/epoch_51
    # ~trainer.wandb_config \
