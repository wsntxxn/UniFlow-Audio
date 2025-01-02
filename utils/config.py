from pathlib import Path
import sys
from typing import Union
import os

import hydra
import omegaconf


def get_warmup_iterations(
    iterations: Union[int, None] = None,
    epochs: Union[float, None] = None,
    epoch_length: Union[int, None] = None
) -> int:
    """
    Derive warmup iterations according to iteration or epoch number.
    """
    if iterations is None and epochs is None:
        raise Exception("Either warmup iterations or epochs must be provided")
    if epochs is None:
        return iterations
    else:
        return int(epoch_length * epochs)


def multiply(*args):
    result = 1
    for arg in args:
        result *= arg
    return result


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolver for hydra configs, which can be used in YAML
    files for dynamically setting values
    """
    omegaconf.OmegaConf.clear_resolvers()
    omegaconf.OmegaConf.register_new_resolver(
        "get_warmup_iterations", get_warmup_iterations, replace=True
    )
    omegaconf.OmegaConf.register_new_resolver("len", len, replace=True)
    omegaconf.OmegaConf.register_new_resolver(
        "multiply", multiply, replace=True
    )


def generate_config_from_command_line_overrides(
    config_file: Union[str, Path]
) -> omegaconf.DictConfig:
    register_omegaconf_resolvers()

    config_file = Path(config_file).absolute()
    config_name = config_file.name.__str__()
    config_path = config_file.parent.__str__()
    config_path = os.path.relpath(config_path, Path(__file__).parent)

    overrides = sys.argv[1:]
    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    omegaconf.OmegaConf.resolve(config)

    return config
