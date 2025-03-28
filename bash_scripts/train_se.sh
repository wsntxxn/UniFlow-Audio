accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml train.py \
    epoch_length=10000 \
    epochs=100 \
    data@data_dict=se \
    train_dataloader.batch_size=1 \
    val_dataloader.batch_size=1 \
    model=diffusion_se \
    exp_dir=se \
    optimizer.lr=5e-5 \
    loss@loss_fn=weighted_sum \
    trainer.wandb_config.project=xtoaudio_se \
    trainer.wandb_config.name=vae_wave_backbone_audioudit_se