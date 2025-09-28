accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml train.py \
    model=flow_matching_tiny \
    model._target_=models.flow_matching.DummyContentAudioFlowMatching \
    model/backbone=input_fusion_dit \
    model.autoencoder.pretrained_ckpt=ckpts/ezaudio_vae/1m.pt \
    exp_name=tiny_dummy_input_fusion \
    exp_dir=experiments/all_visual_sound_no_upsampling/tiny_dummy_input_fusion \
    data@data_dict=train_val_visual_sound \
    epochs=200