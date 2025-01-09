vc submit --image docker.v2.aispeech.com/sjtu/sjtu_wumengyue-xnxpy310:0.0.3 \
    --partition pdgpu-a10 \
    --nopassenv \
    --env HF_HOME="/hpc_stor03/sjtu_home/xuenan.xu/hf_cache" TOKENIZERS_PARALLELISM=false \
    --job x_to_audio_generation \
    --cpu-per-task 8 \
    --mem-per-task 48G \
    --gpu-per-task 8 \
    JOB=1:1 logs/vc/log_a10.JOB.log \
    --cmd "accelerate launch --config_file configs/accelerate/8gpus.yaml train.py \
           warmup_params.warmup_steps=25 \
           train_dataloader.dataset.datasets.0.caption=data/audiocaps/train/caption_toy.jsonl \
           val_dataloader.dataset.datasets.0.caption=data/audiocaps/val/caption_toy.jsonl \
           train_dataloader.batch_size=12 \
           val_dataloader.batch_size=12"