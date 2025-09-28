accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml train.py \
    model=flow_matching_base \
    data/task_sampling@task_sampling_weights=time_align_balanced \
    model._target_=models.flow_matching.DummyContentAudioFlowMatching \
    model/backbone=layer_fusion_dit \
    exp_name=base_dummy_layer_fusion_resume_200k \
    exp_dir=experiments/all_visual_sound_balanced_sampling/base_dummy_layer_fusion_resume_200k \
    data@data_dict=train_val_visual_sound \
    epochs=200 \
    ++trainer.resume_from_checkpoint=experiments/all_visual_sound_balanced_sampling/base_dummy_layer_fusion_resume_200k/checkpoints/step_200000
