# nohup bash bash_scripts_ali/train_dummy_content+cross_attn_adapter_init_8gpus.sh > logs/train_dummy_content+cross_attn_adapter_init_8gpus.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1 
accelerate launch --config_file configs_ali/accelerate/nvidia/8gpus.yaml train.py \
    --config-path configs_ali \
    model=diffusion_dummy_content+cross_attn_adapter \
    data@data_dict=train_init \
    train_dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    epochs=2000 \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    trainer.gradient_accumulation_steps=1 \
    lr_scheduler.name="constant" \
    exp_dir=experiments/DummyContentAudioDiffusion/dummy_content+cross_attn_adapter_init_8gpus \
    trainer.wandb_config.project=DummyContentAudioDiffusion \
    trainer.wandb_config.name=dummy_content+cross_attn_adapter_init_8gpus \