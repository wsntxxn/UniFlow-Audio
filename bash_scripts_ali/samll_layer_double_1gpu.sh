# nohup bash bash_scripts_ali/train_flow_matching_base_dummy_all_1gpu.sh > logs/train_flow_matching_base_dummy_all_1gpu.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

MODEL_SIZE=small #[small,base,large]
BACKBONE=layer # [layer,input]
CONTENT_FUSION=DoubleContent # [DoubleContent,DummyConten,HybridContent,CrossAttention]
GPUS=1gpu
TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1  accelerate launch --config_file configs_ali/accelerate/nvidia/1gpu.yaml train.py \
    --config-path configs_ali \
    exp_name=${MODEL_SIZE}+${BACKBONE}+${CONTENT_FUSION} \
    model=flow_matching_${MODEL_SIZE} \
    model._target_=models.flow_matching.${CONTENT_FUSION}AudioFlowMatching \
    model/backbone=${BACKBONE}_fusion_dit  \
    data@data_dict=train_tts+tta \
    train_dataloader.batch_size=8 \
    train_dataloader.num_workers=2 \
    val_dataloader.batch_size=8 \
    val_dataloader.num_workers=2 \
    epochs=1000 \
    epoch_length=10 \
    max_val_samples=20 \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    lr_scheduler.name="constant" \
    trainer.gradient_accumulation_steps=1 \
    trainer.logger=swanlab \
    exp_dir=experiments/${exp_name}_1gpu \
    +auto_reusme_from_latest_ckpt=true \
    # +cfg_only=true