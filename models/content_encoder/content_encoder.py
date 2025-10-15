from typing import Any
import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        text_encoder: nn.Module = None,
        video_encoder: nn.Module = None,
        midi_encoder: nn.Module = None,
        phoneme_encoder: nn.Module = None,
        pitch_encoder: nn.Module = None,
        audio_encoder: nn.Module = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_encoder = text_encoder
        self.midi_encoder = midi_encoder
        self.phoneme_encoder = phoneme_encoder
        self.pitch_encoder = pitch_encoder
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder

    def encode_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_content_output = []
        batch_content_mask = []
        batch_la_content_output = []

        zero_la_content = torch.zeros(1, 1, self.embed_dim, device=device)
        for content, task in zip(batch_content, batch_task):
            if task == "audio_super_resolution" or task == "speech_enhancement":
                content_dict = {
                    "waveform": torch.as_tensor(content).float(),
                    "waveform_lengths": torch.as_tensor(content.shape[0]),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_output_dict = self.audio_encoder(**content_dict)
                la_content_output_dict = {
                    "output": zero_la_content,
                }
            elif task == "text_to_audio" or task == "text_to_music":
                content_output_dict = self.text_encoder([content])
                la_content_output_dict = {
                    "output": zero_la_content,
                }
            elif task == "video_to_audio":
                content_dict = {
                    "frames": torch.as_tensor(content).float(),
                    "frame_nums": torch.as_tensor(content.shape[0]),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_output_dict = self.video_encoder(**content_dict)
                la_content_output_dict = {
                    "output": zero_la_content,
                }
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
                if "spk" in content:
                    if self.midi_encoder.spk_config.encoding_format == "id":
                        content_dict["spk"] = torch.as_tensor(content["spk"]
                                                             ).long()
                    elif self.midi_encoder.spk_config.encoding_format == "embedding":
                        content_dict["spk"] = torch.as_tensor(content["spk"]
                                                             ).float()
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                content_output_dict = self.midi_encoder(**content_dict)
                la_content_output_dict = {"output": zero_la_content}
            elif task == "text_to_speech":
                content_dict = {
                    "phoneme": torch.as_tensor(content["phoneme"]).long(),
                }
                if "spk" in content:
                    if self.phoneme_encoder.spk_config.encoding_format == "id":
                        content_dict["spk"] = torch.as_tensor(content["spk"]
                                                             ).long()
                    elif self.phoneme_encoder.spk_config.encoding_format == "embedding":
                        content_dict["spk"] = torch.as_tensor(content["spk"]
                                                             ).float()
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                content_output_dict = self.phoneme_encoder(**content_dict)
                la_content_output_dict = {"output": zero_la_content}
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
                content_output_dict = self.pitch_encoder(**content_dict)

                content_dict = {
                    "f0": torch.as_tensor(content["f0"]),
                    "uv": torch.as_tensor(content["uv"]),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                la_content_output_dict = self.pitch_encoder.encode_pitch(
                    **content_dict
                )
            else:
                raise ValueError(f"Unsupported task: {task}")

            batch_content_output.append(content_output_dict["output"][0])
            batch_content_mask.append(content_output_dict["mask"][0])
            batch_la_content_output.append(la_content_output_dict["output"][0])

        batch_content_output = nn.utils.rnn.pad_sequence(
            batch_content_output, batch_first=True, padding_value=0
        )
        batch_content_mask = nn.utils.rnn.pad_sequence(
            batch_content_mask, batch_first=True, padding_value=False
        )
        batch_la_content_output = nn.utils.rnn.pad_sequence(
            batch_la_content_output, batch_first=True, padding_value=0
        )
        return {
            "content": batch_content_output,
            "content_mask": batch_content_mask,
            "length_aligned_content": batch_la_content_output,
        }


class BatchedContentEncoder(ContentEncoder):
    def encode_content(
        self, batch_content: list | dict, batch_task: list[str],
        device: str | torch.device
    ):
        task = batch_task[0]
        zero_la_content = torch.zeros(1, 1, self.embed_dim, device=device)
        if task == "audio_super_resolution" or task == "speech_enhancement":
            content_dict = {
                "waveform":
                    batch_content["content"].unsqueeze(1).float().to(device),
                "waveform_lengths":
                    batch_content["content_lengths"].long().to(device),
            }
            content_output = self.audio_encoder(**content_dict)
            la_content_output = zero_la_content
        elif task == "text_to_audio":
            content_output = self.text_encoder(batch_content)
            la_content_output = zero_la_content
        elif task == "video_to_audio":
            content_dict = {
                "frames":
                    batch_content["content"].float().to(device),
                "frame_nums":
                    batch_content["content_lengths"].long().to(device),
            }
            content_output = self.video_encoder(**content_dict)
            la_content_output = zero_la_content
        elif task == "singing_voice_synthesis":
            content_dict = {
                "phoneme":
                    batch_content["phoneme"].long().to(device),
                "midi":
                    batch_content["midi"].long().to(device),
                "midi_duration":
                    batch_content["midi_duration"].float().to(device),
                "is_slur":
                    batch_content["is_slur"].long().to(device),
                "lengths":
                    batch_content["phoneme_lengths"].long().cpu(),
            }
            if "spk" in batch_content:
                if self.midi_encoder.spk_config.encoding_format == "id":
                    content_dict["spk"] = batch_content["spk"].long(
                    ).to(device)
                elif self.midi_encoder.spk_config.encoding_format == "embedding":
                    content_dict["spk"] = batch_content["spk"].float(
                    ).to(device)
            content_output = self.midi_encoder(**content_dict)
            la_content_output = zero_la_content
        elif task == "text_to_speech":
            content_dict = {
                "phoneme": batch_content["phoneme"].long().to(device),
                "lengths": batch_content["phoneme_lengths"].long().cpu(),
            }
            if "spk" in batch_content:
                if self.phoneme_encoder.spk_config.encoding_format == "id":
                    content_dict["spk"] = batch_content["spk"].long(
                    ).to(device)
                elif self.phoneme_encoder.spk_config.encoding_format == "embedding":
                    content_dict["spk"] = batch_content["spk"].float(
                    ).to(device)
            content_output = self.phoneme_encoder(**content_dict)
            la_content_output = zero_la_content
        elif task == "singing_acoustic_modeling":
            content_dict = {
                "phoneme": batch_content["phoneme"].long().to(device),
                "lengths": batch_content["phoneme_lengths"].long().to(device),
            }
            content_output = self.pitch_encoder(**content_dict)

            content_dict = {
                "f0": batch_content["f0"].float().to(device),
                "uv": batch_content["uv"].float().to(device),
            }
            la_content_output = self.pitch_encoder.encode_pitch(**content_dict)
        else:
            raise ValueError(f"Unsupported task: {task}")

        return {
            "content": content_output["output"],
            "content_mask": content_output["mask"],
            "length_aligned_content": la_content_output,
        }
