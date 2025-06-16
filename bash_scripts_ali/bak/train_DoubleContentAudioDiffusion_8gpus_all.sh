# nohup bash bash_scripts_ali/train_DoubleContentAudioDiffusion_8gpus_tta.sh > logs/train_DoubleContentAudioDiffusion_8gpus_tta.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1 accelerate launch --config_file configs_ali/accelerate/nvidia/8gpus.yaml train.py \
    --config-path configs_ali \
    data@data_dict=train_init \
    train_dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    epochs=1000 \
    model=diffusion_double_content \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    trainer.gradient_accumulation_steps=1 \
    lr_scheduler.name="constant" \
    exp_dir=experiments/DoubleContentAudioDiffusion/init \
    trainer.wandb_config.project=DoubleContentAudioDiffusion \
    trainer.wandb_config.name=init