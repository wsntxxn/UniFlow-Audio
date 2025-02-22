accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml \
    train.py \
    data@data_dict=tta_audiocaps \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12