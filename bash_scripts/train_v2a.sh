accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml train.py \
    warmup_params.warmup_steps=1000 \
    train_dataloader.batch_size=24 \
    val_dataloader.batch_size=24 \
    exp_name=vggsound_token \
    data@data_dict=v2a_vggsound_clip \
    model.content_encoder.video_encoder.video_feat_dim=1024 \
    epoch_length=2000 \
    epochs=100 \
    ~trainer.wandb_config