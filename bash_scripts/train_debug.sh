CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file configs/accelerate/2gpus.yaml \
    train.py \
    train_dataloader.batch_size=1 \
    val_dataloader.batch_size=1 \
    epochs=10 \
    epoch_length=5 \
    exp_dir=experiments/debug \
    gradient_accumulation_steps=1 \
    model.backbone.depth=6 \
    ~trainer.wandb_config
    # +trainer.resume_from_checkpoint=experiments/debug/checkpoints/epoch_5/