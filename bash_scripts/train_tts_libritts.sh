# nohup bash ./bash_scripts/train_tts_libritts_8gpus.sh  > logs/train_tts_libritts_8gpus.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 
accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml train.py \
    data@data_dict=tts_libritts \
    epoch_length=500 \
    epochs=100 \
    train_dataloader.batch_size=8 \
    val_dataloader.batch_size=8 \
    ~trainer.wandb_config \
    exp_name=tts_debug_reference