accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml train.py \
    model=flow_matching_large \
    model._target_=models.flow_matching.HybridContentAudioFlowMatching \
    model/backbone=layer_fusion_dit \
    exp_name=large_hybrid_layer_fusion \
    epochs=250
