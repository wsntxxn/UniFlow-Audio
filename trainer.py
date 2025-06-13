from abc import abstractmethod, ABC
from enum import Enum
from typing import Literal, Any
import shutil
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm import trange, tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import torch.nn as nn
from accelerate.utils import set_seed, broadcast
from accelerate import DistributedDataParallelKwargs
import wandb
from swanlab.integration.accelerate import SwanLabTracker

from utils.accelerate_utilities import AcceleratorSaveTrainableParams


@dataclass
class WandbConfig:
    project: str
    save_dir: str | Path
    name: str
    resume_id: str | None = None


class LRSchedulerInterval(str, Enum):
    EPOCH = "epoch"
    STEP = "step"


class CheckpointMixin(ABC):
    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        ...


@dataclass
class MetricMonitor(CheckpointMixin):

    metric_name: str = "loss"
    mode: Literal["min", "max"] = "min"

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("Mode must be 'min' or 'max'.")
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.worse_count = 0

    def compare(self, x: float, best_x: float) -> bool:
        """Compares the current value with the best value based on mode."""
        return x < best_x if self.mode == "min" else x > best_x

    def __call__(self, metric_dict: dict[str, Any]) -> bool:
        """Checks if the new value is better and updates best_value if so."""
        metric_value = metric_dict[self.metric_name]
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        if self.compare(metric_value, self.best_value):
            self.best_value = metric_value
            self.worse_count = 0
            return True
        self.worse_count += 1
        return False

    def state_dict(self) -> dict:
        """Returns the state of the object as a dictionary."""
        return {
            "mode": self.mode,
            "best_value": self.best_value,
            "worse_count": self.worse_count
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the state from a dictionary."""
        self.mode = state_dict["mode"]
        self.best_value = state_dict["best_value"]
        self.worse_count = state_dict["worse_count"]


@dataclass(kw_only=True)
class Trainer(CheckpointMixin):
    config_dict: dict | None = None
    project_dir: str | Path
    checkpoint_dir: str | Path = None
    logger: str = "wandb"
    wandb_config: WandbConfig | None = None

    train_dataloader: StatefulDataLoader | DataLoader
    val_dataloader: StatefulDataLoader | DataLoader
    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: nn.Module

    epochs: int
    epoch_length: int | None = None
    lr_scheduler_interval: LRSchedulerInterval = LRSchedulerInterval.STEP
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 2.0
    resume_from_checkpoint: str | Path | None = None
    save_every_n_steps: int | None = None
    save_every_n_epochs: int | None = 1
    save_last_k: int | None = 1
    metric_monitor: MetricMonitor | None = None
    early_stop: int | None = None

    def wrap_and_broadcast_value(self, value: Any) -> torch.Tensor:
        value = torch.tensor(value, device=self.accelerator.device)
        broadcast(value, from_process=0)
        return value

    def setup_accelerator(self) -> None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        if self.logger == "swanlab" and self.wandb_config is not None:
            tracker = SwanLabTracker(
                run_name=self.wandb_config.project,
                experiment_name=self.wandb_config.name,
                # logdir=self.wandb_config.save_dir,
            )
        else:
            tracker = self.logger

        self.accelerator = AcceleratorSaveTrainableParams(
            log_with=tracker,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            project_dir=self.project_dir,
            step_scheduler_with_optimizer=(
                self.lr_scheduler_interval == LRSchedulerInterval.STEP
            ),
            kwargs_handlers=[ddp_kwargs]
        )
        # TODO when `loss_fn` does not have named_parameters/buffers, loading will raise error
        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )
        self.accelerator.register_for_checkpointing(self)
        for checkpoint_object in self.checkpoint_objects:
            self.accelerator.register_for_checkpointing(checkpoint_object)
        if self.resume_from_checkpoint is not None:
            self.accelerator.print(
                f"resume from checkpoint: {self.resume_from_checkpoint}"
            )
            self.accelerator.load_state(
                self.resume_from_checkpoint, strict=False
            )

    @abstractmethod
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        ...

    @abstractmethod
    def validation_step(self, batch, batch_idx) -> None:
        ...

    def val_loop(self) -> None:
        self.model.eval()
        torch.set_grad_enabled(False)

        self.on_validation_start()

        pbar = tqdm(
            total=len(self.val_dataloader),
            desc="Validation",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(self.val_dataloader):
            self.validation_step(batch, batch_idx)
            pbar.update()

        pbar.close()

        self.on_validation_end()
        self.model.train()
        torch.set_grad_enabled(True)

    def on_validation_start(self) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def get_val_metrics(self) -> dict[str, Any]:
        return {}

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    @property
    def checkpoint_objects(self) -> list[CheckpointMixin]:
        return []

    def state_dict(self) -> dict:
        state_dict = {"epoch": self.epoch, "step": self.step}
        if isinstance(self.train_dataloader, StatefulDataLoader):
            state_dict["train_dataloader"] = self.train_dataloader.state_dict()
        if self.metric_monitor is not None:
            state_dict["metric_monitor"] = self.metric_monitor.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        if "train_dataloader" in state_dict:
            self.train_dataloader.load_state_dict(
                state_dict["train_dataloader"]
            )
        if "metric_monitor" in state_dict:
            self.metric_monitor.load_state_dict(state_dict["metric_monitor"])

    def clean_checkpoints_to_k(
        self, checkpoints_dir: Path | str, k: int
    ) -> None:
        checkpoints_dir = Path(checkpoints_dir)
        checkpoints = (
            list(checkpoints_dir.glob("epoch_*")) +
            list(checkpoints_dir.glob("step_*"))
        )
        # sort `checkpoints` by their last modified timestamp (ascending order)
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        if k > 0:
            to_delete = checkpoints[:-k] if len(checkpoints) > k else []
        elif k == 0:
            to_delete = checkpoints
        for checkpoint in to_delete:
            shutil.rmtree(checkpoint)

    def save_checkpoint(self, save_dir: Path | str) -> None:
        """
        Note: since `wait_for_everyone` is called, user must be responsible for making sure 
        all processes call or not call this function at the same time!!!
        """
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_dir = Path(save_dir)
            checkpoints_dir = save_dir.parent

            if self.save_last_k:
                self.clean_checkpoints_to_k(
                    checkpoints_dir, self.save_last_k - 1
                )

            self.accelerator.save_state(save_dir)
        self.accelerator.wait_for_everyone()

    def train_loop(self) -> None:
        torch.set_grad_enabled(True)
        self.model.train()
        self.on_train_epoch_start()

        epoch_steps = (self.epoch + 1) * self.epoch_length - self.step

        if self.accelerator.is_main_process:
            range_iterator = trange(
                epoch_steps, desc=f"Epoch {self.epoch + 1}/{self.epochs}"
            )
        else:
            range_iterator = range(epoch_steps)

        for batch_idx in range_iterator:
            try:
                batch = next(self.train_data_iterator)
            except StopIteration:
                self.train_data_iterator = iter(self.train_dataloader)
                batch = next(self.train_data_iterator)

            with self.accelerator.accumulate(self.model):
                loss = self.training_step(batch, batch_idx)
                self.accelerator.log({"train/loss": loss.item()},
                                     step=self.step)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.max_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self.optimizer.step()
                if self.lr_scheduler_interval == LRSchedulerInterval.STEP:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

            self.step += 1

            if self.save_every_n_steps:
                should_save_checkpoint = self.wrap_and_broadcast_value(
                    self.step % self.save_every_n_steps == 0,
                )
                if should_save_checkpoint:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"step_{self.step}"
                    )

        self.val_loop()
        self.epoch += 1

        if self.lr_scheduler_interval == LRSchedulerInterval.EPOCH:
            self.lr_scheduler.step()

        if self.save_every_n_epochs:
            should_save_checkpoint = self.wrap_and_broadcast_value(
                self.epoch % self.save_every_n_epochs == 0
            )
            if should_save_checkpoint:
                self.accelerator.print("\n Saving latest checkpoint...")
                self.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{self.epoch}"
                )

        metric_dict: dict = self.get_val_metrics()
        if self.metric_monitor is not None:
            # save checkpoint if the monitored metric improves
            should_save_checkpoint = self.wrap_and_broadcast_value(
                self.metric_monitor(metric_dict)
            )
            if should_save_checkpoint:
                self.accelerator.print("\n Saving best checkpoint...")
                self.save_checkpoint(self.checkpoint_dir / "best")

            if self.early_stop is not None and self.metric_monitor.worse_count >= self.early_stop:
                self.should_stop_training = True
        else:
            assert self.early_stop is None, "early stop does not have metrics to monitor!"

        # on start of train epoch end func
        self.on_train_epoch_end()

    def on_train_start(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint_dir:
            self.checkpoint_dir = self.project_dir / "checkpoints"
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.accelerator.print(
            f"{self.accelerator.state.num_processes} devices are used in training"
        )

        # if load from previous checkpoint, `epoch` and `step` have been set
        if not hasattr(self, "epoch"):
            self.epoch = 0
        if not hasattr(self, "step"):
            self.step = 0
        self.should_stop_training = False

        # set up `epoch_length` and training data iterator
        if self.epoch_length is None:
            self.epoch_length = len(self.train_dataloader)
        self.train_data_iterator = iter(self.train_dataloader)
        self.val_data_iterator = iter(self.val_dataloader)

        self.accelerator.print(f"training start ............")
        if self.wandb_config is not None:
            self.accelerator.init_trackers(
                self.wandb_config.project,
                init_kwargs={
                    "wandb": {
                        "name": self.wandb_config.name,
                        "dir": self.wandb_config.save_dir,
                        "id": self.wandb_config.resume_id,
                        "resume": "allow",
                    }
                }
            )

    def on_train_end(self) -> None:
        self.accelerator.print(f"training end ............")
        self.accelerator.end_training()
        # wandb sometimes stuck in finishing
        if wandb.run is not None:
            wandb.finish()

    def train(self, seed: int) -> None:
        set_seed(seed)
        self.setup_accelerator()

        self.on_train_start()

        for _ in range(self.epoch, self.epochs):
            self.train_loop()
            if self.should_stop_training:
                break

        self.on_train_end()
