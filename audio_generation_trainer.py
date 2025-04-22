from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
from trainer import Trainer
from utils.logging import LoggingLogger


@dataclass(kw_only=True)
class AudioGenerationTrainer(Trainer):

    logging_file: str | Path

    def on_train_start(self):
        super().on_train_start()
        self.train_loss = 0
        self.train_batch_num = 0
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()
            with open(self.project_dir / "config.yaml", "w") as writer:
                OmegaConf.save(self.config_dict, writer)

        if isinstance(self.model, DistributedDataParallel):
            num_params, trainable_params = self.model.module.count_params()
        else:
            num_params, trainable_params = self.model.count_params()

        if self.accelerator.is_main_process:
            self.logger.info(
                f"parameter number: {num_params}, trainable parameter number: {trainable_params}"
            )

    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_batch_num = 0

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        loss_dict = self.loss_fn(output)
        log_dict = {}
        for loss_name in loss_dict:
            if loss_name != "loss":
                log_dict[f"train/{loss_name}"] = loss_dict[loss_name].item()
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        loss = loss_dict["loss"]
        self.train_loss += loss.item()
        self.train_batch_num += 1
        return loss

    def on_validation_start(self):
        self.val_loss_dict = defaultdict(int)
        self.val_batch_num = 0
        self.val_time_aligned_batch_num = 0

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss_dict = self.loss_fn(output)
        for loss_name in loss_dict:
            self.val_loss_dict[loss_name] += loss_dict[loss_name]
        self.val_batch_num += 1
        if loss_dict["local_duration_loss"].item() != 0.0:
            self.val_time_aligned_batch_num += 1

    def get_val_metrics(self):
        metric_name = self.metric_monitor.metric_name
        assert metric_name in self.val_loss_dict, \
            f"{metric_name} not found in validation loss dict"
        return {
            metric_name: self.val_loss_dict[metric_name] / self.val_batch_num
        }

    def on_train_epoch_end(self):
        train_loss = self.train_loss / self.train_batch_num
        val_loss = self.val_loss_dict["loss"] / self.val_batch_num
        log_dict = {}
        for loss_name in self.val_loss_dict:
            if loss_name == "local_duration_loss":
                log_dict[f"val/{loss_name}"] = self.val_loss_dict[
                    loss_name] / self.val_time_aligned_batch_num
            else:
                log_dict[f"val/{loss_name}"
                        ] = self.val_loss_dict[loss_name] / self.val_batch_num
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        logging_msg = f"epoch[{self.epoch}], train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
        self.accelerator.print(logging_msg)
        if self.accelerator.is_main_process:
            self.logger.info(logging_msg)
