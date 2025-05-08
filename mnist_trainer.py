from pathlib import Path
from dataclasses import dataclass
import random
import datetime
import torch
from torch import nn

import torchvision
from torchvision import transforms
from torchdata.stateful_dataloader import StatefulDataLoader

from trainer import Trainer, WandbConfig, MetricMonitor
from utils.logging import LoggingLogger


class RandomNaNDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, p=0.1):
        self.mnist_dataset = mnist_dataset
        self.p = p

    def __getitem__(self, index):
        feature, label = self.mnist_dataset[index]
        if random.random() < self.p:
            feature = torch.full_like(feature, float("nan"))
        return feature, label

    def __len__(self):
        return len(self.mnist_dataset)


def create_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(
        root="/mnt/cloudstorfs/sjtu_home/xuenan.xu/data/mnist",
        train=True,
        download=True,
        transform=transform
    )
    ds_train = RandomNaNDatasetWrapper(ds_train, p=0.001)
    ds_val = torchvision.datasets.MNIST(
        root="/mnt/cloudstorfs/sjtu_home/xuenan.xu/data/mnist",
        train=False,
        download=True,
        transform=transform
    )

    # dl_train = torch.utils.data.DataLoader(
    dl_train = StatefulDataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    # dl_val = torch.utils.data.DataLoader(
    dl_val = StatefulDataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )
    return dl_train, dl_val


def create_net():
    net = nn.Sequential()
    net.add_module(
        "conv1", nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3)
    )
    net.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
    net.add_module(
        "conv2", nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5)
    )
    net.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
    net.add_module("dropout", nn.Dropout2d(p=0.1))
    net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1)))
    net.add_module("flatten", nn.Flatten())
    net.add_module("linear1", nn.Linear(256, 128))
    net.add_module("relu", nn.ReLU())
    net.add_module("linear2", nn.Linear(128, 10))
    return net


@dataclass(kw_only=True)
class MnistTrainer(Trainer):

    logging_file: str | Path

    def on_train_start(self):
        super().on_train_start()
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()

    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        lr = self.optimizer.param_groups[0]["lr"]
        return loss

    def on_validation_start(self):
        self.validation_stats = {"accurate": 0, "num_elems": 0}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        predictions = preds.argmax(dim=-1)
        output = {"predictions": predictions, "labels": labels}
        output = self.accelerator.gather_for_metrics(output)
        accurate_preds = (output["predictions"] == output["labels"])
        self.validation_stats["accurate"] += accurate_preds.long().sum()
        self.validation_stats["num_elems"] += accurate_preds.shape[0]

    def get_val_metrics(self):
        return {
            "accuracy":
                self.validation_stats["accurate"].item() /
                self.validation_stats["num_elems"]
        }

    def on_validation_end(self):
        eval_metric = self.validation_stats["accurate"].item(
        ) / self.validation_stats["num_elems"]
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.accelerator.print(
            f"epoch[{self.epoch}]@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%"
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                f"epoch[{self.epoch}]@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%"
            )

    def on_train_epoch_end(self):
        if self.accelerator.is_main_process:
            self.logger.info(f"training epoch {self.epoch} ended")


gradient_accumulation_steps = 4

# dl_train, dl_val = create_dataloaders(32 * 32)
dl_train, dl_val = create_dataloaders(64)

model = create_net()
lr = 1e-4
epochs = 10
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=25 * lr,
    epochs=epochs,
    # TODO check the effect of `gradient_accumulation_steps`
    steps_per_epoch=len(dl_train) // gradient_accumulation_steps
)
loss_fn = torch.nn.CrossEntropyLoss()

experiment_dir = "experiments/mnist"

trainer = MnistTrainer(
    project_dir=experiment_dir,
    logging_file=experiment_dir + "/train.log",
    wandb_config=WandbConfig(
        project="test_mnist", save_dir=experiment_dir, name="test1"
    ),
    train_dataloader=dl_train,
    val_dataloader=dl_val,
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    gradient_accumulation_steps=gradient_accumulation_steps,
    loss_fn=loss_fn,
    epochs=epochs,
    save_every_n_epochs=1,
    save_every_n_steps=500,
    save_last_k=3,
    metric_monitor=MetricMonitor(metric_name="accuracy", mode="max")
    # resume_from_checkpoint="experiments/mnist/checkpoints/step_2000"
)
trainer.train(42)
