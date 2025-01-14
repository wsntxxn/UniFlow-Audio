from typing import Any
import math
import copy
from torch.utils.data import DataLoader


def get_warmup_steps(
    dataloader_one_pass_outside_steps: int,
    warmup_steps: int | None = None,
    warmup_epochs: float | None = None,
    epoch_length: int | None = None,
) -> int:
    """
    Derive warmup steps according to step number or epoch number.
    If `warmup_steps` is provided, then just return it. Otherwise, derive
    the warmup steps by epoch length and warmup epoch number.
    """
    if warmup_steps is not None:
        return warmup_steps
    else:
        if epoch_length is None:
            epoch_length = dataloader_one_pass_outside_steps
        assert warmup_epochs is not None, "warmup_steps and warmup_epochs cannot be both None"
        return int(epoch_length * warmup_epochs)


def get_dataloader_one_pass_outside_steps(
    train_dataloader: DataLoader,
    num_processes: int = 1,
):
    """
    dataloader length after DDP, close to `original_length / gpu_number`
    """
    return math.ceil(len(train_dataloader) / num_processes)


def get_total_training_steps(
    train_dataloader: DataLoader,
    epochs: int,
    num_processes: int = 1,
    epoch_length: int | None = None
):
    """
    Calculate the total number of "visible" training steps.

    If `epoch_length` is provided, it is used as the fixed length for each epoch.
    Otherwise, the function will determine the epoch length from `train_dataloader`.

    Args:
        train_dataloader: 
            Training dataloader object.
        epochs: 
            The total number of epochs to run.
        num_processes: 
            The number of parallel processes used for distributed training.
        epoch_length: 
            A fixed number of training steps for each epoch. Defaults to None.

    Returns:
        int: The total number of training steps (i.e., `epochs * epoch_length`).
    """
    # `epoch_length` is not None: fixed length for each epoch
    if epoch_length is None:
        # `epoch_length` is the length of DDP-wrapped `train_dataloader`
        epoch_length = get_dataloader_one_pass_outside_steps(
            train_dataloader, num_processes
        )
    return epochs * epoch_length


def get_dataloader_one_pass_steps_inside_accelerator(
    dataloader_one_pass_steps: int, gradient_accumulation_steps: int,
    num_processes: int
):
    """
    Calculate the number of "visible" training steps for a single pass over the dataloader
    inside an accelerator, accounting for gradient accumulation and distributed training.


    Args:
        dataloader_one_pass_steps:
            The number of steps (batches) in one pass over the dataset.
        gradient_accumulation_steps:
            The number of steps to accumulate gradients before performing a parameter update.
        num_processes:
            The number of parallel processes used for distributed training.

    Returns:
        int: The total number of "visible" training steps for one pass over the dataset,
             multiplied by the number of processes.
    """
    return math.ceil(
        dataloader_one_pass_steps / gradient_accumulation_steps
    ) * num_processes


def get_steps_inside_accelerator_from_outside_steps(
    outside_steps: int, dataloader_one_pass_outside_steps: int,
    dataloader_one_pass_steps_inside_accelerator: int,
    gradient_accumulation_steps: int, num_processes: int
):
    """
    Convert "outside" steps (as observed in wandb logger or similar context) 
    to the corresponding number of "inside" steps (for accelerate lr scheduler).

    Specifically, accelerate lr scheduler call `step()` `num_processes` times for
    every `gradient_accumulation_steps` outside steps.

    Args:
        outside_steps:
            The total number of steps counted outside accelerate context.
        dataloader_one_pass_outside_steps:
            The number of steps (batches) to complete one pass of the dataloader
            outside accelerate.
        dataloader_one_pass_steps_inside_accelerator:
            The number of `lr_scheduler.step()` calls inside accelerate, calculated via
            `get_dataloader_one_pass_steps_inside_accelerator`.
        gradient_accumulation_steps:
            The number of steps to accumulate gradients.
        num_processes:
            The number of parallel processes (GPUs) used in distributed training.

    Returns:
        int: The total number of `lr_scheduler.step()` calls inside accelerate that 
        correspond to the given `outside_steps`.
    """
    num_dataloader_epochs_passed = outside_steps // dataloader_one_pass_outside_steps
    remaining_outside_steps = outside_steps % dataloader_one_pass_outside_steps
    remaining_inside_accelerator_steps = (
        remaining_outside_steps // gradient_accumulation_steps * num_processes
    )
    # accelerate scheduler call `step()` `num_processes` times every
    # `gradient_accumulation_steps` steps:
    # https://github.com/huggingface/accelerate/blob/main/src/accelerate/scheduler.py#L76
    total_steps = (
        num_dataloader_epochs_passed*
        dataloader_one_pass_steps_inside_accelerator +
        remaining_inside_accelerator_steps
    )
    return total_steps


def lr_scheduler_param_adapter(
    config_dict: dict[str, Any], num_training_steps: int, num_warmup_steps: int
) -> dict[str, Any]:
    target_class = config_dict["_target_"]
    return_dict = copy.deepcopy(config_dict)
    if target_class == "transformers.get_scheduler":
        return_dict.update({
            "num_training_steps": num_training_steps,
            "num_warmup_steps": num_warmup_steps
        })

    return return_dict
