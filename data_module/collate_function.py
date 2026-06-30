from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import torch.nn as nn


@dataclass(kw_only=True)
class PaddingCollate:

    pad_keys: list[str] = field(default_factory=lambda: ["waveform"])
    torchify_keys: list[str] = field(default_factory=list)
    time_aligned_tasks: list[str] = field(default_factory=list)
    non_time_aligned_tasks: list[str] = field(default_factory=list)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        collate_samples: dict[str, list[Any]] = {
            k: [dic[k] for dic in batch]
            for k in batch[0]
        }
        batch_keys = list(collate_samples.keys())

        for key in batch_keys:
            if key in self.pad_keys:
                torchified_batch = [
                    torch.as_tensor(d) for d in collate_samples[key]
                ]
                data_batch = nn.utils.rnn.pad_sequence(
                    torchified_batch, batch_first=True
                )
                data_lengths = torch.as_tensor(
                    [len(d) for d in torchified_batch],
                    dtype=torch.int32,
                )

                collate_samples.update({
                    key: data_batch,
                    f"{key}_lengths": data_lengths
                })
            elif key in self.torchify_keys:
                if isinstance(collate_samples[key][0], np.ndarray):
                    collate_samples[key] = np.array(collate_samples[key])
                collate_samples[key] = torch.as_tensor(collate_samples[key])

        if "task" in collate_samples:
            batch_size = len(next(iter(collate_samples.values())))
            time_aligned = torch.zeros(batch_size, 2).bool()
            for i, task in enumerate(collate_samples["task"]):
                if task in self.time_aligned_tasks:
                    time_aligned[i, 0] = True
                if task in self.non_time_aligned_tasks:
                    time_aligned[i, 1] = True
                if task not in self.time_aligned_tasks + self.non_time_aligned_tasks:
                    raise Exception(
                        f"Time align property of {task} is not defined!"
                    )
            collate_samples["time_aligned"] = time_aligned

        return collate_samples


@dataclass(kw_only=True)
class PaddingCollateWithAnyContent(PaddingCollate):
    content_pad_keys: list[str] = field(default_factory=list)
    content_torchify_keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.content_collate_fn = PaddingCollate(
            pad_keys=self.content_pad_keys,
            torchify_keys=self.content_torchify_keys
        )

    def __call__(self, batch):
        batch = super().__call__(batch)
        content = batch["content"]
        if isinstance(content[0], dict):
            content = self.content_collate_fn(content)
        elif isinstance(content[0],
                        torch.Tensor) or isinstance(content[0], np.ndarray):
            content = [torch.as_tensor(d) for d in content]
            padded_content = nn.utils.rnn.pad_sequence(
                content, batch_first=True
            )
            content_lengths = torch.as_tensor(
                [len(d) for d in content],
                dtype=torch.int32,
            )
            content = {
                "content": padded_content,
                "content_lengths": content_lengths,
            }
        batch.update({"content": content})
        return batch
