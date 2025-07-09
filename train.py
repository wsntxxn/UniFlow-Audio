from pathlib import Path
from copy import deepcopy
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import hydra
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.state import PartialState

from utils.config import register_omegaconf_resolvers
from utils.lr_scheduler_utilities import (
    get_warmup_steps, get_dataloader_one_pass_outside_steps,
    get_total_training_steps, get_steps_inside_accelerator_from_outside_steps,
    get_dataloader_one_pass_steps_inside_accelerator,
    lr_scheduler_param_adapter
)
from models.common import CountParamsBase
from trainer import Trainer
from copy import deepcopy

register_omegaconf_resolvers()


def setup_dataloader_args(config: dict):
    dataloader_config = deepcopy(config)
    args = {}
    if "sampler" in config:
        data_source = hydra.utils.instantiate(
            config["dataset"], _convert_="all"
        )
        sampler = hydra.utils.instantiate(
            config["sampler"], data_source=data_source, _convert_="all"
        )
        args["sampler"] = sampler
        dataloader_config.pop("sampler")
    elif "batch_sampler" in config:
        data_source = hydra.utils.instantiate(
            config["dataset"], _convert_="all"
        )
        batch_sampler = hydra.utils.instantiate(
            config["batch_sampler"], data_source=data_source, _convert_="all"
        )
        args["batch_sampler"] = batch_sampler
        dataloader_config.pop("batch_sampler")
    return args, dataloader_config


def setup_resume_cfg(config):
    if "resume_from_checkpoint" in config["trainer"]:
        ckpt_dir = Path(config["trainer"]["resume_from_checkpoint"])
        exp_dir = ckpt_dir.parent.parent
        resumed_config = OmegaConf.load(exp_dir / "config.yaml")
        resumed_config["trainer"].update({
            "resume_from_checkpoint": ckpt_dir.__str__(),
            "logging_config": config["trainer"]
                              ["logging_config"],  # for resume wandb runs
        })
    elif config.get("auto_reusme_from_latest_ckpt", False):
        exp_dir = Path(config["exp_dir"])
        ckpt_root = exp_dir / "checkpoints"
        if ckpt_root.is_dir() and any(p.is_dir() for p in ckpt_root.iterdir()):
            # use last ckpt
            ckpt_dir: Path = sorted((exp_dir / "checkpoints").iterdir())[-1]
            resumed_config = OmegaConf.load(exp_dir / "config.yaml")
            resumed_config["trainer"].update({
                "resume_from_checkpoint": ckpt_dir.__str__(),
                "logging_config": config["trainer"]
                                  ["logging_config"],  # for resume wandb runs
            })
        else:
            resumed_config = config
    else:
        resumed_config = config
    if "resume_from_checkpoint" in resumed_config["trainer"]:
        print(
            f'\n train will resume from checkpoint: {resumed_config["trainer"]["resume_from_checkpoint"]}\n '
        )
    else:
        print('\n train will start from scratch\n ')
    return resumed_config


def main():

    configs = []

    @hydra.main(version_base=None, config_path="configs", config_name="train")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    if config.get("cfg_only", False):
        with open("./config.yaml", "w") as f:
            OmegaConf.save(config, f)
            print(f'config.yaml saved to {f.name}')
            return

    config = setup_resume_cfg(config)

    # helper state for accessing information about the current training environment
    state = PartialState()

    model: CountParamsBase = hydra.utils.instantiate(config["model"])
    train_data_args, train_dataloader_config = setup_dataloader_args(
        config["train_dataloader"]
    )
    train_dataloader = hydra.utils.instantiate(
        train_dataloader_config, **train_data_args, _convert_="all"
    )
    val_data_args, val_dataloader_config = setup_dataloader_args(
        config["val_dataloader"]
    )
    val_dataloader = hydra.utils.instantiate(
        val_dataloader_config, **val_data_args, _convert_="all"
    )
    optimizer = hydra.utils.instantiate(
        config["optimizer"], params=model.parameters(), _convert_="all"
    )

    # `accelerator.prepare` is very confusing for multi-gpu, gradient accumulation scenario:
    # For more information: see https://github.com/huggingface/diffusers/issues/4387,
    # https://github.com/huggingface/diffusers/issues/9633, and
    # https://github.com/huggingface/diffusers/issues/3954
    dataloader_one_pass_outside_steps = get_dataloader_one_pass_outside_steps(
        train_dataloader, state.num_processes
    )
    total_training_steps = get_total_training_steps(
        train_dataloader, config["epochs"], state.num_processes,
        config["epoch_length"]
    )
    dataloader_one_pass_steps_inside_accelerator = (
        get_dataloader_one_pass_steps_inside_accelerator(
            dataloader_one_pass_outside_steps,
            config["gradient_accumulation_steps"], state.num_processes
        )
    )
    num_training_updates = get_steps_inside_accelerator_from_outside_steps(
        total_training_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], state.num_processes
    )

    num_warmup_steps = get_warmup_steps(
        **config["warmup_params"],
        dataloader_one_pass_outside_steps=dataloader_one_pass_outside_steps
    )
    num_warmup_updates = get_steps_inside_accelerator_from_outside_steps(
        num_warmup_steps, dataloader_one_pass_outside_steps,
        dataloader_one_pass_steps_inside_accelerator,
        config["gradient_accumulation_steps"], state.num_processes
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
