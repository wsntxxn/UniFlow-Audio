accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml \
    train.py \
    exp_name=audiocaps_dummy_content_fm \
    model=flow_matching \
    data@data_dict=tta_audiocaps \
    warmup_params.warmup_steps=1000 \
    epoch_length=Null \
    epochs=100 \
    ~trainer.wandb_config
