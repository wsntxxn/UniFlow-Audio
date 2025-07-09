EPOCH=74
CKPT_DIR=experiments/small_hybrid_layer_fusion/checkpoints/epoch_${EPOCH}
EXP_DIR=$(realpath $CKPT_DIR/../..)
INFER_DIR="infer_epoch${EPOCH}"
export PYTHONPATH=.

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=tts_libritts \
    infer_args.guidance_scale=5.0 \
    +data_dict.libritts.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=svs_m4singer \
    infer_args.guidance_scale=5.0 \
    +data_dict.m4singer.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=tta_audiocaps \
    infer_args.guidance_scale=5.0 \
    +data_dict.audiocaps.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=se \
    infer_args.guidance_scale=5.0 \
    +data_dict.voicebank_demand.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=audio_sr \
    infer_args.guidance_scale=5.0 \
    +data_dict.ttshq.test.max_samples=20 \
    +data_dict.musdb.test.max_samples=20 \
    +data_dict.moises.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=ttm_msd \
    infer_args.guidance_scale=5.0 \
    +data_dict.msd.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python inference.py \
    +ckpt_dir=${CKPT_DIR} \
    data@data_dict=v2a_vggsound_clip \
    infer_args.guidance_scale=5.0 \
    +data_dict.vggsound_clip.test.max_samples=20 \
    infer_args.num_steps=20 \
    wav_dir=${INFER_DIR}

python generate_postprocess/merge_v2a_audio_video.py \
    --aid_video_mapping data/vggsound/mapping.csv \
    --audio_path ${EXP_DIR}/${INFER_DIR}/video_to_audio/ \
    --output_dir ${EXP_DIR}/${INFER_DIR}/video_to_audio_video
