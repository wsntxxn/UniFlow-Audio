N_LAYERS=24
D_MODEL=1024

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs_ali/accelerate/nvidia/1gpu.yaml train.py \
    train_dataloader.batch_size=12 \
    val_dataloader.batch_size=12 \
    data@data_dict=svs_opencpop \
    model.backbone.embed_dim=${D_MODEL} \
    model.backbone.depth=${N_LAYERS} \
    exp_dir=experiments/wave_dit_ada_sola_bias_diff/opencpop \
    trainer.wandb_config.project=singing_opencpop \
    trainer.wandb_config.name=wave_dit_ada_sola_bias_diff