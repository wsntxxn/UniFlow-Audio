from pathlib import Path
import sys
import os

import hydra
import omegaconf
from omegaconf import OmegaConf


def multiply(*args):
    result = 1
    for arg in args:
        result *= arg
    return result


def get_pitch_downsample_ratio(
    autoencoder_config: dict, pitch_frame_resolution: float
):
    latent_frame_resolution = autoencoder_config[
        "downsampling_ratio"] / autoencoder_config["sample_rate"]
    return round(latent_frame_resolution / pitch_frame_resolution)


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolver for hydra configs, which can be used in YAML
    files for dynamically setting values
    """
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("len", len, replace=True)
    OmegaConf.register_new_resolver("multiply", multiply, replace=True)
    OmegaConf.register_new_resolver(
        "get_pitch_downsample_ratio", get_pitch_downsample_ratio, replace=True
    )


def generate_config_from_command_line_overrides(
    config_file: str | Path
) -> omegaconf.DictConfig:
    register_omegaconf_resolvers()

    config_file = Path(config_file).resolve()
    config_name = config_file.name.__str__()
    config_path = config_file.parent.__str__()
    config_path = os.path.relpath(config_path, Path(__file__).resolve().parent)

    overrides = sys.argv[1:]
    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    omegaconf.OmegaConf.resolve(config)

    return config
