import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.config import register_omegaconf_resolvers
from utils.lr_scheduler_utilities import (
    get_warmup_steps, get_dataloader_one_pass_outside_steps,
    get_total_training_steps, get_steps_inside_accelerator_from_outside_steps,
    get_dataloader_one_pass_steps_inside_accelerator,
    lr_scheduler_param_adapter
)
from models.common import CountParamsBase
from trainer import Trainer

register_omegaconf_resolvers()


def main():

    configs = []

    @hydra.main(
        version_base=None, config_path="configs", config_name="default"
    )
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    # helper instance for accessing information about the current training environment
    helper_accelerator = Accelerator()

    model: CountParamsBase = hydra.utils.instantiate(config["model"])
    train_dataloader = hydra.utils.instantiate(
        config["train_dataloader"], _convert_="all"
    )
    val_dataloader = hydra.utils.instantiate(
        config["val_dataloader"], _convert_="all"
    )
    optimizer = hydra.utils.instantiate(
        config["optimizer"], params=model.parameters(), _convert_="all"
    )

    # `accelerator.prepare` is very confusing for multi-gpu, gradient accumulation scenario:
    # For more information: see https://github.com/huggingface/diffusers/issues/4387,
    # https://github.com/huggingface/diffusers/issues/9633, and
    # https://github.com/huggingface/diffusers/issues/3954
    dataloader_one_pass_outside_steps = get_dataloader_one_pass_outside_steps(
        train_dataloader, helper_accelerator.num_processes
    )
    total_training_steps = get_total_training_steps(
        train_dataloader, config["epochs"], helper_accelerator.num_processes,
        config["epoch_length"]
    )
    dataloader_one_pass_steps_inside_accelerator = (
        get_dataloader_one_pass_steps_inside_accelerator(
            dataloader_one_pass_outside_steps,
            config["gradient_accumulation_steps"],
            helper_accelerator.num_processes
        )
    )
    num_training_updates = get_steps_inside_accelerator_from_outside_steps(
        total_training_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], helper_accelerator.num_processes
    )

    num_warmup_steps = get_warmup_steps(
        **config["warmup_params"],
        dataloader_one_pass_outside_steps=dataloader_one_pass_outside_steps
    )
    num_warmup_updates = get_steps_inside_accelerator_from_outside_steps(
        num_warmup_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], helper_accelerator.num_processes
    )

    lr_scheduler_config = lr_scheduler_param_adapter(
        config_dict=config["lr_scheduler"],
        num_training_steps=num_training_updates,
        num_warmup_steps=num_warmup_updates
    )

    lr_scheduler = hydra.utils.instantiate(
        lr_scheduler_config, optimizer=optimizer, _convert_="all"
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
    trainer.config_dict = config  # assign here, don't instantiate it
    trainer.train(seed=config["seed"])


if __name__ == "__main__":
    main()
