ckpt_dir=experiments/large_dummy_layer_fusion/iters_400000
num_steps=20
iters=400000
infer_dir="infer_iters_${iters}_steps_${num_steps}"
export PYTHONPATH=.

num_samples=50

# Parse command line arguments
__snapshot_before=$(mktemp)
declare -p > "$__snapshot_before"

while [[ $# -gt 0 ]]; do
    key="$1"
    val="$2"

    if [[ "$key" =~ ^--(.+) ]]; then
        var_name="${BASH_REMATCH[1]}"

        if [[ -n "$val" && ! "$val" =~ ^-- ]]; then
            if grep -q -E "^declare .* $var_name=" "$__snapshot_before"; then
                eval "$var_name=\"\$val\""
            fi
            shift 2
        else
            shift 1
        fi
    else
        shift 1
    fi
done

rm -f "$__snapshot_before"

python inference.py \
    ckpt_dir_or_file=${ckpt_dir} \
    data@data_dict=test_cfg \
    infer_args.guidance_scale=5.0 \
    infer_args.num_steps=${num_steps} \
    wav_dir=${infer_dir} \
    max_test_samples=${num_samples}

python inference.py \
    ckpt_dir_or_file=${ckpt_dir} \
    data@data_dict=test_no_cfg \
    infer_args.guidance_scale=1.0 \
    infer_args.num_steps=${num_steps} \
    wav_dir=${infer_dir} \
    max_test_samples=${num_samples}


python generate_postprocess/merge_v2a_audio_video.py \
    --aid_video_mapping data/vggsound/mapping.csv \
    --audio_path ${exp_dir}/${infer_dir}/video_to_audio/ \
    --output_dir ${exp_dir}/${infer_dir}/video_to_audio_video \
    --backend moviepy
