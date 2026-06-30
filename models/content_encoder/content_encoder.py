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
        audio_encoder: nn.Module = None,
        speaker_encoder: nn.Module = None,
        singer_encoder: nn.Module = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_encoder = text_encoder
        self.midi_encoder = midi_encoder
        self.phoneme_encoder = phoneme_encoder
        self.pitch_encoder = pitch_encoder
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.speaker_encoder = speaker_encoder
        self.singer_encoder = singer_encoder

    def encode_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_ta_content_output = []
        batch_ta_content_mask = []
        batch_nta_content_output = []
        batch_nta_content_mask = []
        batch_la_content_output = []

        dummy_ta_content = torch.zeros(1, self.embed_dim, device=device)
        dummy_ta_content_mask = torch.tensor([True],
                                             dtype=torch.bool,
                                             device=device)
        dummy_nta_content = torch.zeros(1, self.embed_dim, device=device)
        dummy_nta_content_mask = torch.tensor([True],
                                              dtype=torch.bool,
                                              device=device)
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
                out_dict = self.audio_encoder(**content_dict)
                la_content_output_dict = {
                    "output": zero_la_content,
                }
                ta_content = out_dict["output"][0]
                ta_content_mask = out_dict["mask"][0]
                nta_content = dummy_nta_content
                nta_content_mask = dummy_nta_content_mask
            elif task == "text_to_audio" or task == "text_to_music":
                out_dict = self.text_encoder([content])
                ta_content = dummy_ta_content
                ta_content_mask = dummy_ta_content_mask
                nta_content = out_dict["output"][0]
                nta_content_mask = out_dict["mask"][0]
                la_content_output_dict = {
                    "output": zero_la_content,
                }
            elif task == "video_to_audio":
                content_dict = {
                    "clip": torch.as_tensor(content["clip"]).float(),
                    "sync": torch.as_tensor(content["sync"]).float(),
                    "duration": torch.as_tensor([content["duration"]]).float(),
                    "clip_lengths": torch.as_tensor(content["clip"].shape[0]).long(),
                    "sync_lengths": torch.as_tensor(content["sync"].shape[0]).long(),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                out_dict = self.video_encoder(**content_dict)
                ta_content = out_dict["ta_output"][0]
                ta_content_mask = out_dict["ta_mask"][0]
                if out_dict["nta_output"] is None:
                    nta_content = dummy_nta_content
                    nta_content_mask = dummy_nta_content_mask
                else:
                    nta_content = out_dict["nta_output"][0]
                    nta_content_mask = out_dict["nta_mask"][0]

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
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                out_dict = self.midi_encoder(**content_dict)
                ta_content = out_dict["output"][0]
                ta_content_mask = out_dict["mask"][0]
                out_dict = self.singer_encoder(
                    speaker_id=torch.as_tensor(content["spk"]).long().unsqueeze(0).to(device),
                )
                nta_content = out_dict["output"][0]
                nta_content_mask = out_dict["mask"][0]
                la_content_output_dict = {"output": zero_la_content}
            elif task == "text_to_speech":
                content_dict = {
                    "phoneme": torch.as_tensor(content["phoneme"]).long(),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["phoneme_lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                out_dict = self.phoneme_encoder(**content_dict)
                ta_content = out_dict["output"][0]
                ta_content_mask = out_dict["mask"][0]
                prompt_waveform = torch.as_tensor(
                    content["prompt_waveform"]
                ).float().to(device).unsqueeze(0)
                prompt_waveform_lengths = torch.as_tensor([
                    len(content["prompt_waveform"])
                ])
                out_dict = self.speaker_encoder(
                    **{
                        "waveform": prompt_waveform,
                        "waveform_lengths": prompt_waveform_lengths,
                    }
                )
                nta_content = out_dict["output"][0]
                nta_content_mask = out_dict["mask"][0]
                la_content_output_dict = {"output": zero_la_content}
            else:
                raise ValueError(f"Unsupported task: {task}")

            batch_ta_content_output.append(ta_content)
            batch_ta_content_mask.append(ta_content_mask)
            batch_nta_content_output.append(nta_content)
            batch_nta_content_mask.append(nta_content_mask)
            batch_la_content_output.append(la_content_output_dict["output"][0])

        batch_ta_content_output = nn.utils.rnn.pad_sequence(
            batch_ta_content_output, batch_first=True, padding_value=0
        )
        batch_ta_content_mask = nn.utils.rnn.pad_sequence(
            batch_ta_content_mask, batch_first=True, padding_value=False
        )
        batch_nta_content_output = nn.utils.rnn.pad_sequence(
            batch_nta_content_output, batch_first=True, padding_value=0
        )
        batch_nta_content_mask = nn.utils.rnn.pad_sequence(
            batch_nta_content_mask, batch_first=True, padding_value=False
        )
        batch_la_content_output = nn.utils.rnn.pad_sequence(
            batch_la_content_output, batch_first=True, padding_value=0
        )
        return {
            "ta_content": batch_ta_content_output,
            "ta_content_mask": batch_ta_content_mask,
            "nta_content": batch_nta_content_output,
            "nta_content_mask": batch_nta_content_mask,
            "la_content": batch_la_content_output,
        }


class BatchedContentEncoder(ContentEncoder):
    def encode_content(
        self, batch_content: list | dict, batch_task: list[str],
        device: str | torch.device
    ):
        task = batch_task[0]
        batch_size = len(batch_task)

        dummy_ta_content = torch.zeros(1, 1, self.embed_dim, device=device)
        dummy_ta_content_mask = torch.tensor([[True]],
                                             dtype=torch.bool,
                                             device=device)
        dummy_nta_content = torch.zeros(1, 1, self.embed_dim, device=device)
        dummy_nta_content_mask = torch.tensor([[True]],
                                              dtype=torch.bool,
                                              device=device)
        zero_la_content = torch.zeros(1, 1, self.embed_dim, device=device)

        if task == "audio_super_resolution" or task == "speech_enhancement":
            content_dict = {
                "waveform":
                    batch_content["content"].unsqueeze(1).float().to(device),
                "waveform_lengths":
                    batch_content["content_lengths"].long().to(device),
            }
            out_dict = self.audio_encoder(**content_dict)
            ta_content_output = out_dict["output"]
            ta_content_mask = out_dict["mask"]
            nta_content_output = dummy_nta_content.expand(batch_size, -1, -1)
            nta_content_mask = dummy_nta_content_mask.expand(batch_size, -1)
            la_content_output = zero_la_content
        elif task == "text_to_audio" or task == "text_to_music":
            out_dict = self.text_encoder(batch_content)
            ta_content_output = dummy_ta_content.expand(batch_size, -1, -1)
            ta_content_mask = dummy_ta_content_mask.expand(batch_size, -1)
            nta_content_output = out_dict["output"]
            nta_content_mask = out_dict["mask"]
            la_content_output = zero_la_content
        elif task == "video_to_audio":
            out_dict = self.video_encoder(**batch_content)
            ta_content_output = out_dict["ta_output"]
            ta_content_mask = out_dict["ta_mask"]
            nta_content_output = out_dict["nta_output"]
            nta_content_mask = out_dict["nta_mask"]
            if nta_content_output is None:
                nta_content_output = dummy_nta_content.expand(
                    batch_size, -1, -1
                )
                nta_content_mask = dummy_nta_content_mask.expand(
                    batch_size, -1
                )
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

            out_dict = self.midi_encoder(**content_dict)
            ta_content_output = out_dict["output"]
            ta_content_mask = out_dict["mask"]

            out_dict = self.singer_encoder(
                speaker_id=batch_content["spk"].long().to(device)
            )
            nta_content_output = out_dict["output"]
            nta_content_mask = out_dict["mask"]
            la_content_output = zero_la_content
        elif task == "text_to_speech":
            content_dict = {
                "phoneme":
                    batch_content["phoneme"].long().to(device),
                "phoneme_lengths":
                    batch_content["phoneme_lengths"].long().cpu(),
            }
            out_dict = self.phoneme_encoder(**content_dict)
            ta_content_output = out_dict["output"]
            ta_content_mask = out_dict["mask"]
            out_dict = self.speaker_encoder(
                waveform=batch_content["prompt_waveform"].float().to(device),
                waveform_lengths=batch_content["prompt_waveform_lengths"].long(
                ),
            )
            nta_content_output = out_dict["output"]
            nta_content_mask = out_dict["mask"]
            la_content_output = zero_la_content
        elif task == "singing_acoustic_modeling":
            content_dict = {
                "phoneme": batch_content["phoneme"].long().to(device),
                "lengths": batch_content["phoneme_lengths"].long().to(device),
            }
            out_dict = self.pitch_encoder(**content_dict)
            ta_content_output = out_dict["output"]
            ta_content_mask = out_dict["mask"]
            nta_content_output = dummy_nta_content.expand(batch_size, -1, -1)
            nta_content_mask = dummy_nta_content_mask.expand(batch_size, -1)
            content_dict = {
                "f0": batch_content["f0"].float().to(device),
                "uv": batch_content["uv"].float().to(device),
            }
            la_content_output = self.pitch_encoder.encode_pitch(**content_dict)
        else:
            raise ValueError(f"Unsupported task: {task}")

        return {
            "ta_content": ta_content_output,
            "ta_content_mask": ta_content_mask,
            "nta_content": nta_content_output,
            "nta_content_mask": nta_content_mask,
            "la_content": la_content_output,
        }
