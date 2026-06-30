from omegaconf import OmegaConf

# def get_pitch_downsample_ratio(
#     autoencoder_config: dict, pitch_frame_resolution: float
# ):
#     latent_frame_resolution = autoencoder_config[
#         "downsampling_ratio"] / autoencoder_config["sample_rate"]
#     return round(latent_frame_resolution / pitch_frame_resolution)


def multiply(*args):
    res = 1
    for arg in args:
        res *= arg
    return res


def get_latent_token_rate(sample_rate: int, downsampling_ratio: int):
    return sample_rate // downsampling_ratio


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolver for hydra configs, which can be used in YAML
    files for dynamically setting values
    """
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("multiply", multiply, replace=True)
    OmegaConf.register_new_resolver(
        "get_latent_token_rate", get_latent_token_rate, replace=True
    )


def resolve_dot_key(config: dict, dot_key: str):
    keys = dot_key.split(".")
    last_cfg = config
    for key in keys[:-1]:
        try:
            last_cfg = last_cfg[key]
        except KeyError:
            raise KeyError(
                f"Key {key} not found in config when resolving {dot_key}."
            )
    return last_cfg, keys[-1]
