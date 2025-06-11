from typing import Any
import numpy as np
import torch
import torch.nn as nn


class PaddingCollate:
    def __init__(
        self,
        pad_keys: list[str] = ["waveform"],
        torchify_keys: list[str] = []
    ):
        self.pad_keys = pad_keys
        self.torchify_keys = torchify_keys

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

        return collate_samples


class PaddingCollateWithAnyContent(PaddingCollate):
    def __init__(
        self,
        pad_keys: list[str] = ["waveform"],
        torchify_keys: list[str] = [],
        content_pad_keys: list[str] = [],
        content_torchify_keys: list[str] = []
    ):
        super().__init__(pad_keys, torchify_keys)
        self.content_collate_fn = PaddingCollate(
            content_pad_keys, content_torchify_keys
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
