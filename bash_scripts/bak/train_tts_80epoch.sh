# nohup bash ./bash_scripts/train_tts_80epoch.sh  > logs/train_tts_80epoch.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/1gpu.yaml train.py \
    epoch_length=1000 \
    epochs=80 \
    train_dataloader.batch_size=96 \
    val_dataloader.batch_size=96 \
    model=diffusion_tts \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1.5e-4 \
    loss_fn.weights.local_duration_loss=0.1 \
    trainer.wandb_config.project=tts \
    exp_dir=experiments/tts/ljspeech/VaribleLengthAudioDiffusion_24Khz_lr1.5e-4_80epoch \
    trainer.wandb_config.name=ljspeech-VaribleLengthAudioDiffusion_24Khz_lr1.5e-4_80epoch \
    +trainer.resume_from_checkpoint=experiments/tts/ljspeech/VaribleLengthAudioDiffusion_24Khz_lr1.5e-4/checkpoints/epoch_20
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/opencpop/checkpoints/epoch_51
    # ~trainer.wandb_config \
    # train_data_dict.data_list=data.opencpop.train \
    # val_data_dict.data_list=data.opencpop.val \
    # test_data_dict.data_list=data.opencpop.test \
