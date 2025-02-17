CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/1gpu.yaml train.py \
    epoch_length=1000 \
    epochs=100 \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    model=diffusion_singing \
    loss@loss_fn=weighted_sum \
    exp_dir=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/popcs \
    optimizer.lr=5e-5 \
    loss_fn.weights.local_duration_loss=0.0 \
    trainer.wandb_config.project=singing_popcs \
    trainer.wandb_config.name=vae_wave_backbone_audioudit_layers_24_dim_1024_loss_diff
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/opencpop/checkpoints/epoch_51
    # ~trainer.wandb_config \
