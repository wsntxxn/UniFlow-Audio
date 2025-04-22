# nohup bash ./bash_scripts/train_tts_libritts.sh  > logs/train_tts_libritts.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/1gpu.yaml train.py \
    epoch_length=1000 \
    epochs=100 \
    train_dataloader.batch_size=24 \
    val_dataloader.batch_size=24 \
    gradient_accumulation_steps=4 \
    model=diffusion_tts \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1.5e-4 \
    lr_scheduler.name=constant \
    loss_fn.weights.local_duration_loss=0.1 \
    trainer.wandb_config.project=tts \
    trainer.wandb_config.name=librits_wo_conv_xvector \
    exp_dir=experiments/tts/librits/wo_conv_xvector \
    # +trainer.resume_from_checkpoint=experiments/waveform_vae_audioudit_layers_24_dim_1024_diffusion/opencpop/checkpoints/epoch_51
    # ~trainer.wandb_config \
    # train_data_dict.data_list=data.opencpop.train \
    # val_data_dict.data_list=data.opencpop.val \
    # test_data_dict.data_list=data.opencpop.test \
