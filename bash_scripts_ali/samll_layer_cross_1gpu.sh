# nohup bash bash_scripts_ali/train_flow_matching_base_dummy_all_1gpu.sh > logs/train_flow_matching_base_dummy_all_1gpu.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

MODEL_SIZE=small #[small,base,large]
BACKBONE=udit # [layer,input]
CONTENT_FUSION=CrossAttention # [DoubleContent,DummyContent,HybridContent,CrossAttention]
GPUS=1gpu
EXP_NAME="${MODEL_SIZE}+${BACKBONE}+${CONTENT_FUSION}+${GPUS}"
TORCH_NCCL_TRACE_BUFFER_SIZE=67108864 NCCL_DEBUG=INFO  NCCL_ASYNC_ERROR_HANDLING=1  accelerate launch --config_file configs_ali/accelerate/nvidia/${GPUS}.yaml train.py \
    --config-path configs_ali \
    exp_name=${EXP_NAME} \
    model=flow_matching_${MODEL_SIZE} \
    model._target_=models.flow_matching.${CONTENT_FUSION}AudioFlowMatching \
    model/backbone=udit  \
    data@data_dict=train_all \
    train_dataloader.batch_size=24 \
    train_dataloader.num_workers=2 \
    val_dataloader.batch_size=24 \
    val_dataloader.num_workers=2 \
    epochs=250 \
    epoch_length=2000 \
    max_val_samples=500 \
    loss@loss_fn=weighted_sum \
    optimizer.lr=5e-5 \
    lr_scheduler.name="linear" \
    trainer.gradient_accumulation_steps=1 \
    trainer.logger=swanlab \
    exp_dir="experiments/${MODEL_SIZE}/${EXP_NAME}" \
    +auto_reusme_from_latest_ckpt=true \
    ~model.backbone.ta_context_dim \
    # +cfg_only=true \