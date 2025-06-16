# nohup bash bash_scripts_ali/train_flow_matching_base_dummy_all_1gpu.sh > logs/train_flow_matching_base_dummy_all_1gpu.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1  accelerate launch --config_file configs_ali/accelerate/nvidia/1gpu.yaml train.py \
    --config-path configs_ali \
    model=flow_matching_base_dummy \
    data@data_dict=train_wottm \
    train_dataloader.batch_size=4 \
    val_dataloader.batch_size=4 \
    epochs=1000 \
    epoch_length=10 \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    lr_scheduler.name="constant" \
    trainer.gradient_accumulation_steps=1 \
    exp_dir=experiments/flow_matching_base/flow_matching_base_dummy_init_1gpu \