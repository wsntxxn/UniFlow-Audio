#!/bin/bash

# ------------------------------
infer_dir="experiments/base_hybrid_layer_fusion/infer_eval_guidance_5.0_steps_20_iters_240000"
exp_name="base_hybrid_layer_fusion"
v2a_only=0
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

if [ "${v2a_only}" -eq 0 ]; then
    $PYTHONBIN evaluation/tts.py \
        --audio_dir ${infer_dir}/text_to_speech/ \
        --libritts_txt_dir /cpfs02/shared/speechllm/data/LibriTTS \
        --xp_name $exp_name \
        --ref_transcript_path data/libritts/test/ref_transcription.json \
        --ref_audio_path data/libritts/test/ref_audio.json \
        --output_path ${infer_dir}/tts_results.txt

    $PYTHONBIN evaluation/svs.py \
        --ref_audio_jsonl data/m4singer/test/audio.jsonl \
        --gen_audio_dir ${infer_dir}/singing_voice_synthesis/ \
        --output_file ${infer_dir}/svs_results.txt

    $PYTHONBIN evaluation/t2a.py \
        --ref_audio_jsonl data/audiocaps_v2/test/audio_renamed.jsonl \
        -rc data/audiocaps_v2/test/caption.jsonl \
        -gd ${infer_dir}/text_to_audio/ \
        -o ${infer_dir}/t2a_results.txt

    $PYTHONBIN evaluation/se.py \
        --ref_dir /oss-speechllm-a100/xuxuenan/speech_enhancement/voicebank+demand/clean_testset_wav/ \
        --gen_dir ${infer_dir}/speech_enhancement/ \
        --uuid_jsonl /oss-speechllm-a100/xuxuenan/speech_enhancement/voicebank+demand/test_metadata_audio.jsonl \
        --output_file ${infer_dir}/se_results.txt

    bash bash_scripts/eval_audio_sr.sh ${infer_dir}

    $PYTHONBIN evaluation/t2a.py \
        --ref_audio_jsonl data/music_caps/audio_renamed.jsonl \
        -rc data/music_caps/caption.jsonl \
        -gd ${infer_dir}/text_to_music/ \
        -o ${infer_dir}/t2m_cnn14_results.txt
fi

$PYTHONBIN evaluation/v2a.py \
    -ra data/visual_sound/test_audio_16000Hz_0s_to_10.0s.jsonl \
    -ibv data/visual_sound/ib_visual_embed.h5 \
    -syncv data/visual_sound/test_videos_fps_25_sr_16000.jsonl \
    -gd ${infer_dir}/video_to_audio/ \
    -o ${infer_dir}/v2a_results.txt
