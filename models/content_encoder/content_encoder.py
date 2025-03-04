from typing import Any
import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(
        self,
        text_encoder: nn.Module,
        midi_encoder: nn.Module = None,
        pitch_encoder: nn.Module = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.midi_encoder = midi_encoder
        self.pitch_encoder = pitch_encoder

    def encode_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_output = []
        batch_mask = []

        for content, task in zip(batch_content, batch_task):
            if task == "text_to_audio":
                output_dict = self.text_encoder([content])
            elif task == "singing_voice_synthesis":
                content_dict = {
                    "phoneme":
                        torch.as_tensor(content["phoneme"]).long(),
                    "midi":
                        torch.as_tensor(content["midi"]).long(),
                    "midi_duration":
                        torch.as_tensor(content["midi_duration"]).float(),
                    "is_slur":
                        torch.as_tensor(content["is_slur"]).long()
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                output_dict = self.midi_encoder(**content_dict)
            elif task == "singing_acoustic_modeling":
                content_dict = {
                    "phoneme": torch.as_tensor(content["phoneme"]).long(),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                output_dict = self.pitch_encoder(**content_dict)

            batch_output.append(output_dict["output"][0])
            batch_mask.append(output_dict["mask"][0])

        batch_output = nn.utils.rnn.pad_sequence(
            batch_output, batch_first=True, padding_value=0
        )
        batch_mask = nn.utils.rnn.pad_sequence(
            batch_mask, batch_first=True, padding_value=False
        )
        return batch_output, batch_mask

    def encode_time_aligned_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_output = []

        for content, task in zip(batch_content, batch_task):
            if task == "text_to_audio":
                output_dict = {"output": torch.zeros(1, 1, device=device)}
            elif task == "singing_voice_synthesis":
                output_dict = {"output": torch.zeros(1, 1, device=device)}
            elif task == "singing_acoustic_modeling":
                content_dict = {
                    "f0": torch.as_tensor(content["f0"]),
                    "uv": torch.as_tensor(content["uv"]),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                output_dict = self.pitch_encoder.encode_pitch(**content_dict)

            batch_output.append(output_dict["output"][0])

        batch_output = nn.utils.rnn.pad_sequence(
            batch_output, batch_first=True, padding_value=0
        )
        return batch_output
