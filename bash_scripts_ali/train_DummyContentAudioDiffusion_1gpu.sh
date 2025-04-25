# nohup bash bash_scripts/train_DummyContentAudioDiffusion_1gpu.sh > logs/train_DummyContentAudioDiffusion_1gpu.log 2>&1  &
echo "Current time: $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')" 
echo "Running script name: $(basename "$0")" 

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml train.py \
    --config_dir configs_ali \
    warmup_params.warmup_steps=2000 \
    data@data_dict=train_init \
    train_dataloader.batch_size=8 \
    val_dataloader.batch_size=24 \
    epochs=1000 \
    model=diffusion \
    loss@loss_fn=weighted_sum \
    optimizer.lr=1e-4 \
    trainer.gradient_accumulation_steps=1 \
    lr_scheduler.name="constant" \
    exp_dir=experiments/DummyContentAudioDiffusion/init \
    trainer.wandb_config.project=DummyContentAudioDiffusion \
    trainer.wandb_config.name=init