from typing import Any
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
                data_batch = torch.nn.utils.rnn.pad_sequence(
                    torchified_batch, batch_first=True
                )
                data_lengths = torch.as_tensor([
                    len(d) for d in torchified_batch
                ],
                                               dtype=torch.int32)

                collate_samples.update({
                    key: data_batch,
                    f"{key}_lengths": data_lengths
                })
            elif key in self.torchify_keys:
                collate_samples[key] = torch.as_tensor(collate_samples[key])

        return collate_samples
