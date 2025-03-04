N_LAYERS=20
D_MODEL=256

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml train.py \
    epoch_length=1000 \
    epochs=100 \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    model=diffusion_singing \
    model.backbone.depth=20 \
    exp_dir=experiments/waveform_vae_audioudit_layers_${N_LAYERS}_dim_${D_MODEL}_diffusion/popcs_lr1e-3 \
    optimizer.lr=1e-3 \
    loss@loss_fn=weighted_sum \
    ~trainer.wandb_config \
    loss_fn.weights.local_duration_loss=0.0
    # trainer.wandb_config.project=singing_popcs \
    # trainer.wandb_config.name=vae_wave_backbone_audio_udit_layers_${N_LAYERS}_dim_${D_MODEL}_loss_diff \
