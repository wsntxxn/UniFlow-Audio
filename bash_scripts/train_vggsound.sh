accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml train.py \
    warmup_params.warmup_steps=1000 \
    model=flow_matching \
    train_dataloader.batch_size=24 \
    val_dataloader.batch_size=24 \
    exp_name=vggsound_dummy_content_fm \
    data@data_dict=v2a_vggsound_clip \
    epoch_length=2000 \
    epochs=100 \
    trainer.logger=wandb \
    ~trainer.wandb_config \
    +trainer.resume_from_checkpoint=experiments/vggsound_dummy_content_fm/checkpoints/epoch_53