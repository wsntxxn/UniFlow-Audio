from copy import deepcopy

import hydra
from accel_hydra.train_launcher import TrainLauncher

from utils.config import register_omegaconf_resolvers


def init_dataloader_from_config(config: dict):
    config = deepcopy(config)
    kwargs = {}
    dataset = hydra.utils.instantiate(config.pop("dataset"), _convert_="all")
    kwargs["dataset"] = dataset
    if "sampler" in config:
        sampler_cfg = config.pop("sampler")
        sampler_cls = hydra.utils.get_class(sampler_cfg.pop("_target_"))
        sampler = sampler_cls(data_source=dataset, **sampler_cfg)
        kwargs["sampler"] = sampler
    elif "batch_sampler" in config:
        batch_sampler_cfg = config.pop("batch_sampler")
        batch_sampler_cls = hydra.utils.get_class(
            batch_sampler_cfg.pop("_target_")
        )
        if "sampler" in batch_sampler_cfg:
            sampler_cfg = batch_sampler_cfg.pop("sampler")
            sampler_cls = hydra.utils.get_class(sampler_cfg.pop("_target_"))
            sampler = sampler_cls(data_source=dataset, **sampler_cfg)
            batch_sampler = batch_sampler_cls(
                sampler=sampler, **batch_sampler_cfg
            )
        else:
            batch_sampler = batch_sampler_cls(
                data_source=dataset, **batch_sampler_cfg
            )
        kwargs["batch_sampler"] = batch_sampler

    if "collate_fn" in config:
        collate_fn = hydra.utils.instantiate(
            config.pop("collate_fn"), _convert_="all"
        )
    else:
        collate_fn = None
    kwargs["collate_fn"] = collate_fn
    dataloader_cls = hydra.utils.get_class(config.pop("_target_"))
    dataloader = dataloader_cls(**kwargs, **config)
    return dataloader


class Launcher(TrainLauncher):
    @staticmethod
    def get_register_resolver_fn():
        return register_omegaconf_resolvers

    def get_dataloaders(self):
        train_dataloader = init_dataloader_from_config(
            self.config["train_dataloader"]
        )
        if "val_dataloader" in self.config and self.config["val_dataloader"
                                                          ] is not None:
            val_dataloader = init_dataloader_from_config(
                self.config["val_dataloader"]
            )
        else:
            val_dataloader = None
        return train_dataloader, val_dataloader
