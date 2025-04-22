# nohup bash bash_scripts/train_tts_tta_8gpus_constant.sh > logs/train_tts_tta_8gpus_constant.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 
TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 accelerate launch --config_file configs/accelerate/8gpus.yaml train.py \
    epoch_length=1000 \
    epochs=80 \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    gradient_accumulation_steps=1 \
    model=diffusion_multitask_1024d \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    lr_scheduler.name=linear \
    loss_fn.weights.local_duration_loss=0.05 \
    loss_fn.weights.global_duration_loss=0.05 \
    trainer.wandb_config.project=tts \
    trainer.wandb_config.name=train_tts_tta_constant3 \
    exp_dir=experiments/tts_tta/libritts_audiocaps_constant3 \
    # +trainer.resume_from_checkpoint=experiments/tts_tta/libritts_audiocaps/checkpoints/epoch_12
    # ~trainer.wandb_config \
    # train_data_dict.data_list=data.opencpop.train \
    # val_data_dict.data_list=data.opencpop.val \
    # test_data_dict.data_list=data.opencpop.test \