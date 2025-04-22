# nohup bash ./bash_scripts/train_tta_1gpu.sh  > logs/train_tta_1gpu.log 2>&1  &
CUDA_VISIBLE_DEVICES=0 TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1 \
accelerate launch --config_file configs/accelerate/1gpu.yaml train.py \
    train_dataloader.batch_size=72 \
    val_dataloader.batch_size=72 \
    data@data_dict=tta \
    epochs=200 \
    epoch_length=1000 \
    gradient_accumulation_steps=1 \
    model=diffusion \
    loss@loss_fn=identity \
    optimizer.lr=1e-4 \
    lr_scheduler.name=linear \
    trainer.wandb_config.project=tts \
    trainer.wandb_config.name=audiocaps_linear_1gpu \
    trainer.metric_monitor.metric_name=loss \
    exp_dir=experiments/tta/audiocaps/audiocaps_linear_1gpu \
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_udit_diffusion_epoch200_lr1e-4_broken/audiocaps/checkpoints/epoch_20