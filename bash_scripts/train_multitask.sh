accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml train.py \
    model=flow_matching_small \
    model._target_=models.flow_matching.HybridContentAudioFlowMatching \
    model/backbone=input_fusion_dit \
    exp_name=small_hybrid_input_fusion
