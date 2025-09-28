accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml train.py \
    model=flow_matching_tiny \
    model._target_=models.flow_matching.DummyContentAudioFlowMatching \
    model/backbone=layer_fusion_dit \
    exp_name=tiny_dummy_layer_fusion \
    exp_dir=experiments/all_visual_sound_balanced_sampling/tiny_dummy_layer_fusion \
    data@data_dict=train_val_visual_sound \
    epochs=200
