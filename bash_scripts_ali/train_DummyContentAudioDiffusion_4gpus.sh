# nohup bash bash_scripts_ali/train_DummyContentAudioDiffusion_4gpus.sh > logs/train_DummyContentAudioDiffusion_4gpus.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml train.py \
    --config_dir configs_ali \
    data@data_dict=train_init \
    train_dataloader.batch_size=16 \
    val_dataloader.batch_size=16 \
    epochs=1000 \
    model=diffusion \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    trainer.gradient_accumulation_steps=1 \
    lr_scheduler.name="constant" \
    exp_dir=experiments/DummyContentAudioDiffusion/init \
    trainer.wandb_config.project=DummyContentAudioDiffusion \
    trainer.wandb_config.name=init