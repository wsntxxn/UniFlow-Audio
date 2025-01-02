from typing import Any
import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def encode_content(self, batch_content: list[Any], batch_task: list[str]):
        batch_output = []
        batch_mask = []

        for content, task in zip(batch_content, batch_task):
            if task == "text_to_audio":
                output_dict = self.text_encoder([content])
                batch_output.append(output_dict["output"][0])
                batch_mask.append(output_dict["mask"][0])

        batch_output = nn.utils.rnn.pad_sequence(
            batch_output, batch_first=True, padding_value=0
        )
        batch_mask = nn.utils.rnn.pad_sequence(
            batch_mask, batch_first=True, padding_value=False
        )
        return batch_output, batch_mask
