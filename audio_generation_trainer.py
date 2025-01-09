from dataclasses import dataclass
import datetime
from pathlib import Path
import yaml

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
                yaml.dump(self.config_dict, writer)

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        self.train_loss += loss.item()
        self.train_batch_num += 1
        return loss

    def on_validation_start(self):
        self.val_loss = 0
        self.val_batch_num = 0

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch)
        self.val_loss += loss.item()
        self.val_batch_num += 1

    def on_train_epoch_end(self):
        train_loss = self.train_loss / self.train_batch_num
        val_loss = self.val_loss / self.val_batch_num
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging_msg = f"{now_time} @ epoch[{self.epoch}], train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
        self.accelerator.print(logging_msg)
        if self.accelerator.is_main_process:
            self.logger.info(logging_msg)
