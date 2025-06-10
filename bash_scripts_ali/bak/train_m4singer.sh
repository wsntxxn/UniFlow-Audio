N_LAYERS=24
D_MODEL=1024


CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs_ali/accelerate/nvidia/1gpu.yaml train.py \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    data@data_dict=svs_m4singer \
    model=diffusion \
    model.backbone.embed_dim=${D_MODEL} \
    model.backbone.depth=${N_LAYERS} \
    exp_name=noise_weight \
    trainer.wandb_config.project=singing_m4singer \
    # ~trainer.wandb_config \
