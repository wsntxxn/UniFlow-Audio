# nohup bash bash_scripts_ali/train_DummyContentAudioDiffusion_32gpus_tts+tta.sh > logs/train_DummyContentAudioDiffusion_32gpus_tts+tta.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "RANK=$RANK"
TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1 
accelerate launch --config_file configs_ali/accelerate/nvidia/32gpus.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank ${RANK} \
    train.py \
    --config-path configs_ali \
    data@data_dict=train_tts+tta \
    train_dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    epochs=1000 \
    model=diffusion \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    trainer.gradient_accumulation_steps=1 \
    lr_scheduler.name="constant" \
    exp_dir=experiments/DummyContentAudioDiffusion/tts+tta_32gpus \
    trainer.wandb_config.project=DummyContentAudioDiffusion \
    trainer.wandb_config.name=tts+tta_32gpus