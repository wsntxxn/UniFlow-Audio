#!/bin/bash

# ------------------------------
exp_dir=experiments/base_hybrid_layer_fusion
ckpt_dir=experiments/base_hybrid_layer_fusion/epoch_150_ckpt
num_steps=25
iters=300000
# ------------------------------


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

echo "exp_dir: $exp_dir"
echo "ckpt_dir: $ckpt_dir"
echo "num_steps: $num_steps"


num_gpus=$(nvidia-smi -L | wc -l)

infer_dir="infer_eval_steps_${num_steps}_iters_${iters}"
export PYTHONPATH=.
accelerate launch --config-file configs/accelerate/nvidia/8gpus.yaml \
    --num_processes $num_gpus \
    inference.py \
    data@data_dict=test_no_cfg \
    exp_dir=${exp_dir} \
    ckpt_dir=${ckpt_dir} \
    infer_args.guidance_scale=1.0 \
    infer_args.num_steps=${num_steps} \
    wav_dir=${infer_dir}

accelerate launch --config-file configs/accelerate/nvidia/8gpus.yaml \
    --num_processes $num_gpus \
    inference.py \
    data@data_dict=test_cfg \
    exp_dir=${exp_dir} \
    ckpt_dir=${ckpt_dir} \
    infer_args.guidance_scale=5.0 \
    infer_args.num_steps=${num_steps} \
    wav_dir=${infer_dir}