vc submit --image docker.v2.aispeech.com/sjtu/sjtu_wumengyue-xnxpy310:0.0.3 \
    --partition pdgpu-a10 \
    --nopassenv \
    --env HF_HOME="/hpc_stor03/sjtu_home/xuenan.xu/hf_cache" TOKENIZERS_PARALLELISM=false \
    --job x_to_audio_generation \
    --cpu-per-task 8 \
    --mem-per-task 48G \
    --gpu-per-task 4 \
    JOB=1:1 logs/vc/log_a10.JOB.log \
    --cmd "accelerate launch --config_file configs/accelerate/4gpus.yaml train.py \
           train_dataloader.batch_size=12 \
           val_dataloader.batch_size=12 \
           exp_dir=experiments/waveform_vae_udit_diffusion_epoch100_lr1e-4/audiocaps"