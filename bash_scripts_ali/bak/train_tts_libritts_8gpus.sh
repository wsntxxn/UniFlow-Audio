# nohup bash ./bash_scripts_ali/train_tts_libritts_8gpus.sh  > logs/train_tts_libritts_8gpus.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 
accelerate launch --config_file configs/accelerate/8gpus.yaml train.py \
    data@data_dict=tts \
    epoch_length=1000 \
    epochs=200 \
    train_dataloader.batch_size=24 \
    val_dataloader.batch_size=24 \
    gradient_accumulation_steps=1 \
    model=diffusion_tts \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1.5e-4 \
    lr_scheduler.name=linear \
    loss_fn.weights.local_duration_loss=0.1 \
    trainer.wandb_config.project=tts \
    trainer.wandb_config.name=librits_wo_conv_xvector_8gpus_resume_epoch_123 \
    exp_dir=experiments/tts/librits/wo_conv_xvector_8gpus_resume_epoch_123 \
    # +trainer.resume_from_checkpoint=experiments/tts/librits/wo_conv_xvector_8gpus/checkpoints/epoch_123
    # ~trainer.wandb_config \
    # train_data_dict.data_list=data.opencpop.train \
    # val_data_dict.data_list=data.opencpop.val \
    # test_data_dict.data_list=data.opencpop.test \
