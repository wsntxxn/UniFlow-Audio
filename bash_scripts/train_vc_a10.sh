vc submit --image docker.v2.aispeech.com/sjtu/sjtu_wumengyue-xnxpy310:0.0.3 \
    --partition pdgpu-a10 \
    --nopassenv \
    --env HF_HOME="/hpc_stor03/sjtu_home/xuenan.xu/hf_cache" TOKENIZERS_PARALLELISM=false \
    --job x_to_audio_generation \
    --cpu-per-task 4 \
    --mem-per-task 24G \
    --gpu-per-task 1 \
    JOB=1:1 logs/vc/log_a10.JOB.log \
    --cmd "bash bash_scripts/train_tta.sh"