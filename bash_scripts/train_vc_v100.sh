vc submit --image docker.v2.aispeech.com/sjtu/sjtu_wumengyue-xnxpy310:0.0.3 \
    --partition pdgpu-v100 \
    --nopassenv \
    --env HF_HOME="/hpc_stor03/sjtu_home/xuenan.xu/hf_cache" TOKENIZERS_PARALLELISM=false \
    --job x_to_audio_generation \
    --cpu-per-task 8 \
    --mem-per-task 48G \
    --gpu-per-task 4 \
    JOB=1:1 logs/vc/log_v100.JOB.log \
    --cmd "bash bash_scripts/train_tta.sh"