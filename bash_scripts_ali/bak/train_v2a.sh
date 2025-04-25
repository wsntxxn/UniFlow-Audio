accelerate launch --config_file configs/accelerate/nvidia/1gpus.yaml train.py \
    warmup_params.warmup_steps=2000 \
    train_dataloader.batch_size=8 \
    train_dataloader.num_workers=8 \
    val_dataloader.batch_size=20 \
    model=diffusion_v2a \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    lr_scheduler.name="linear" \
    exp_dir=experiments/v2a-vggsound-cavp-debug \
    epochs=2 \
    trainer.gradient_accumulation_steps=1 \
    trainer.wandb_config.project=x2audio_v2a-debug \
    trainer.wandb_config.name=v2a_cavp 