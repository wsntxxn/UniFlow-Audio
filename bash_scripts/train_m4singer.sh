accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml \
    train.py \
    exp_name=m4singer_dummy_content_fm \
    data@data_dict=svs_m4singer \
    model=flow_matching_base \
    warmup_params.warmup_steps=1000 \
    epoch_length=500 \
    epochs=100