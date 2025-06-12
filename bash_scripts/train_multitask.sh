accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml train.py \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    model=flow_matching_small \
    model._target_=models.flow_matching.HybridContentAudioFlowMatching \
    model/backbone=input_fusion_dit \
    exp_name=small_hybrid_input_fusion
