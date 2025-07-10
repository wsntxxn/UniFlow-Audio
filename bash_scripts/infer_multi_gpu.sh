accelerate launch --config-file configs/accelerate/nvidia/4gpus.yaml \
    inference_multi_gpu.py \
    +exp_dir=experiments/small_hybrid_input_fusion/ \
    data@data_dict=tta_audiocaps \
    infer_args.guidance_scale=5.0 \
    infer_args.num_steps=20