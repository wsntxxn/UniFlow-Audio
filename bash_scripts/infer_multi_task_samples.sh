EXP_DIR=experiments/large_hybrid_layer_fusion/
NUM_STEPS=20
INFER_DIR="infer_epoch_last_guidance_5.0_steps_${NUM_STEPS}"
export PYTHONPATH=.

NUM_SAMPLES=50

python inference.py \
    +exp_dir=${EXP_DIR} \
    +use_best=False \
    infer_args.guidance_scale=5.0 \
    infer_args.num_steps=${NUM_STEPS} \
    wav_dir=${INFER_DIR} \
    max_test_samples=${NUM_SAMPLES}


python generate_postprocess/merge_v2a_audio_video.py \
    --aid_video_mapping data/vggsound/mapping.csv \
    --audio_path ${EXP_DIR}/${INFER_DIR}/video_to_audio/ \
    --output_dir ${EXP_DIR}/${INFER_DIR}/video_to_audio_video \
    --backend moviepy
