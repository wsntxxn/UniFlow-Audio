#!/bin/bash

# ------------------------------
infer_dir="experiments/base_hybrid_layer_fusion/infer_eval_guidance_5.0_steps_20_iters_240000"
exp_name="base_hybrid_layer_fusion"
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

echo "eval_all.sh: Running with the following parameters:"
echo "infer_dir: $infer_dir"
echo "exp_name: $exp_name"

PYTHONBIN="/cpfs04/user/xuxuenan/miniconda3/envs/gen_eval/bin/python"
export CLAP_MODEL_PATH="/cpfs02/shared/speechllm/xuxuenan/hf_cache/hub/models--lukewys--laion_clap/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/630k-audioset-best.pt"

export PYTHONPATH=.:evaluation/GMELab

$PYTHONBIN evaluation/tts.py \
    --audio_dir ${infer_dir}/text_to_speech/ \
    --libritts_txt_dir /cpfs02/shared/speechllm/LibriTTS \
    --xp_name $exp_name \
    --ref_transcript_path data/libritts/test/ref_transcription.json \
    --ref_audio_path data/libritts/test/ref_audio.json

$PYTHONBIN evaluation/svs.py \
    --ref_audio_jsonl data/m4singer/test/audio.jsonl \
    --gen_audio_dir ${infer_dir}/singing_voice_synthesis/ \
    --output_file evaluation/result/svs_results_${exp_name}.jsonl

$PYTHONBIN evaluation/tta.py \
    --ref_audio_jsonl data/audiocaps_v2/test/audio_renamed.jsonl \
    -rc data/audiocaps_v2/test/caption.jsonl \
    -gd ${infer_dir}/text_to_audio/ \
    -o evaluation/result/tta_results_${exp_name}.jsonl

$PYTHONBIN evaluation/se.py \
    --ref_dir /oss-speechllm-a100/xuxuenan/speech_enhancement/voicebank+demand/clean_testset_wav/ \
    --gen_dir ${infer_dir}/speech_enhancement/ \
    --uuid_jsonl /oss-speechllm-a100/xuxuenan/speech_enhancement/voicebank+demand/test_metadata_audio.jsonl \
    --output_file evaluation/result/se_results_${exp_name}.jsonl

bash bash_scripts/eval_audio_sr.sh ${infer_dir}

$PYTHONBIN evaluation/tta.py \
    --ref_audio_jsonl data/music_caps/audio_renamed.jsonl \
    -rc data/music_caps/caption.jsonl \
    -gd ${infer_dir}/text_to_music/ \
    -o evaluation/result/ttm_cnn14_results_${exp_name}.jsonl

$PYTHONBIN evaluation/v2a.py \
    -ra data/vggsound/test_audio_16000Hz_0s_to_10.0s.jsonl \
    -ibv data/vggsound/test_videos.jsonl \
    -syncv data/vggsound/test_videos_fps_25_sr_16000.jsonl \
    -gd ${infer_dir}/video_to_audio/ \
    -o evaluation/result/v2a_results_${exp_name}.txt
