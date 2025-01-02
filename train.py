import logging
import json

import hydra
from omegaconf import OmegaConf, DictConfig

from utils.config import register_omegaconf_resolvers
from trainer import Trainer

logger = logging.Logger(__file__)

register_omegaconf_resolvers()


def main():

    configs = []

    @hydra.main(config_path="configs", config_name="default")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    logger.info("config: ")
    logger.info(json.dumps(config, indent=2))

    model = hydra.utils.instantiate(config["model"])
    train_dataloader = hydra.utils.instantiate(
        config["train_dataloader"], _convert_="all"
    )
    val_dataloader = hydra.utils.instantiate(
        config["val_dataloader"], _convert_="all"
    )
    optimizer = hydra.utils.instantiate(
        config["optimizer"], params=model.parameters(), _convert_="all"
    )
    lr_scheduler = hydra.utils.instantiate(
        config["lr_scheduler"], optimizer=optimizer, _convert_="all"
    )
    loss_fn = hydra.utils.instantiate(config["loss_fn"], _convert_="all")

    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        _convert_="all"
    )
    trainer.config_dict = config,  # assign here, don't instantiate it
    trainer.train(seed=config["seed"])


if __name__ == "__main__":
    main()
