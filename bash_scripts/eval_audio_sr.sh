#!/bin/bash

INFER_DIR=experiments/small_double_layer_fusion/infer_eval_test

if [ $# -eq 1 ]; then
    INFER_DIR=$1
fi

echo "INFER_DIR: $INFER_DIR"

python generate_postprocess/make_audio_jsonl.py \
    -d $INFER_DIR/audio_super_resolution/ \
    -t sr \
    -o $INFER_DIR/sr.jsonl 

cat $INFER_DIR/sr.jsonl | grep "esc" > $INFER_DIR/sr_esc.jsonl
cat $INFER_DIR/sr.jsonl | grep "vctk" > $INFER_DIR/sr_vctk.jsonl
cat $INFER_DIR/sr.jsonl | grep "musdb" > $INFER_DIR/sr_musdb.jsonl

python evaluation/sr.py \
    -r /oss-speechllm-a100/xuxuenan/audio_super_resolution/musdb/test/audio.jsonl \
    -gj $INFER_DIR/sr_musdb.jsonl \
    -o $INFER_DIR/sr_musdb_res.txt \
    -rb /oss-speechllm-a100/xuxuenan/audio_super_resolution/musdb/ \
    -c 8

python evaluation/sr.py \
    -r /oss-speechllm-a100/xuxuenan/audio_super_resolution/vctk_test/test/audio.jsonl \
    -gj $INFER_DIR/sr_vctk.jsonl \
    -o $INFER_DIR/sr_vctk_res.txt \
    -rb /oss-speechllm-a100/xuxuenan/audio_super_resolution/vctk_test/ \
    -c 8

python evaluation/sr.py \
    -r /oss-speechllm-a100/xuxuenan/audio_super_resolution/esc_test/test/audio.jsonl \
    -gj $INFER_DIR/sr_esc.jsonl \
    -o $INFER_DIR/sr_esc_res.txt \
    -rb /oss-speechllm-a100/xuxuenan/audio_super_resolution/esc_test/test \
    -c 8

for file in $INFER_DIR/sr_musdb_res.txt $INFER_DIR/sr_vctk_res.txt $INFER_DIR/sr_esc_res.txt; do
    echo "===> $(basename $file) <==="
    cat $file
    echo ""
done > $INFER_DIR/sr_all_res.txt