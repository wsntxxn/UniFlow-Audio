accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml train.py \
    model=flow_matching_tiny \
    model._target_=models.flow_matching.CrossAttentionAudioFlowMatching \
    model/backbone=udit \
    exp_name=tiny_cross \
    exp_dir=experiments/tiny_cross \
    ~model.backbone.ta_context_dim \
    data@data_dict=train_val_visual_sound \
    epochs=200
