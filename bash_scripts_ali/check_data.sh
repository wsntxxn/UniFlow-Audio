# nohup bash bash_scripts_ali/check_data.sh > logs/check_data.log 2>&1  &

echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

python check_nan.py \
    --config-path configs_ali \
    data@data_dict=all_except_tts+tta
    # data@data_dict=VCTK+Demand
    # train_dataloader.batch_size=16 \
    # val_dataloader.batch_size=16 \
    # epochs=1000 \
    # model=diffusion_double_content \
    # loss@loss_fn=weighted_sum \
    # optimizer.lr=1e-4 \
    # trainer.gradient_accumulation_steps=1 \
    # lr_scheduler.name="constant" \
    # exp_dir=experiments/DoubleContentAudioDiffusion/tta_audiocaps_2gpus \
    # trainer.wandb_config.project=DoubleContentAudioDiffusion \
    # trainer.wandb_config.name=tta_audiocaps_2gpus