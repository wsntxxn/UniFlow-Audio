accelerate launch --config_file configs/accelerate/4gpus.yaml \
    train.py \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    epochs=200 \
    exp_dir=experiments/waveform_vae_udit_diffusion_epoch200_lr1e-4/audiocaps
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_udit_diffusion_epoch200_lr1e-4_broken/audiocaps/checkpoints/epoch_20