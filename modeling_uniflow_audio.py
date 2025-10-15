from typing import Any, Sequence
from pathlib import Path
import json
import shutil

import h5py
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors.torch import load_file
import hydra
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5EncoderModel, T5Tokenizer


class UniFlowAudioModel(nn.Module):
    def __init__(self, model_name: str = "wsntxxn/UniFlow-Audio-large"):
        assert model_name in (
            "wsntxxn/UniFlow-Audio-large",
            "wsntxxn/UniFlow-Audio-medium",
            "wsntxxn/UniFlow-Audio-small",
        )
        super().__init__()
        model_dir = snapshot_download(repo_id=model_name)
        model_dir = Path(model_dir)
        self.config = OmegaConf.load(model_dir / "config.yaml")
        self.config["model"]["autoencoder"]["pretrained_ckpt"] = str(
            model_dir / self.config["model"]["autoencoder"]["pretrained_ckpt"]
        )
        self.model = hydra.utils.instantiate(
            self.config["model"], _convert_="all"
        )
        state_dict = load_file(model_dir / "model.safetensors")
        self.model.load_pretrained(state_dict)
        self.model.eval()

        self.g2p_model_path = model_dir / "mfa_g2p" / "english_us_arpa_unhashed.zip"
        if not self.g2p_model_path.exists():
            ori_model_path = (model_dir / "mfa_g2p" /
                              "english_us_arpa.zip").resolve()
            shutil.copy(ori_model_path, self.g2p_model_path)

        self.tts_phone_set_path = model_dir / "mfa_g2p" / "phone_set.json"
        self.build_tts_phone_mapping()
        self.svs_phone_set_path = model_dir / "svs" / "phone_set.json"
        singers = json.load(open(model_dir / "svs" / "spk_set.json", "r"))
        self.svs_singer_mapping = {
            singer: i
            for i, singer in enumerate(singers)
        }
        self.svs_pinyin2ph = model_dir / "svs" / "m4singer_pinyin2ph.txt"

        self.task_to_instructions = {}
        with h5py.File(model_dir / "instructions" / "t5_embeddings.h5") as hf:
            for key in hf.keys():
                self.task_to_instructions[key] = hf[key][()]

        self.init_instruction_encoder()

    def build_tts_phone_mapping(self):
        with open(self.tts_phone_set_path, "r", encoding="utf-8") as f:
            phone_set = json.load(f)

        self.tts_phone2id = {p: i for i, p in enumerate(phone_set)}

    def init_instruction_encoder(self):
        self.instruction_tokenizer = T5Tokenizer.from_pretrained(
            "google/flan-t5-large"
        )
        self.instruction_encoder = T5EncoderModel.from_pretrained(
            "google/flan-t5-large"
        )
        self.instruction_encoder.eval()

    @torch.inference_mode()
    def encode_instruction(self, instruction: list[str], device: torch.device):
        with torch.amp.autocast(enabled=False):
            tokens = self.instruction_tokenizer(
                instruction,
                max_length=self.instruction_tokenizer.model_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)
            output = self.instruction_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            output = output.last_hidden_state
            length = attention_mask.sum(dim=1)
            return output, length

    @torch.inference_mode()
    def sample(
        self,
        content: list[Any],
        task: list[str],
        is_time_aligned: Sequence[bool],
        instruction: list[str] | None = None,
        instruction_idx: list[int] | None = None,
        num_steps: int = 20,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
    ):
        device = self.model.dummy_param.device

        if instruction is None:
            instructions = []
            instruction_lengths = []
            for sample_idx, task_ in enumerate(task):
                if instruction_idx:
                    instruction_idx_ = instruction_idx[sample_idx]
                else:
                    instruction_idx_ = 0
                instruction_ = self.task_to_instructions[
                    f"{task_}_{instruction_idx_}"]
                instructions.append(torch.as_tensor(instruction_))
                instruction_lengths.append(instruction_.shape[0])
            instructions = pad_sequence(instructions,
                                        batch_first=True).to(device)
            instruction_lengths = torch.as_tensor(instruction_lengths
                                                 ).to(device)
        else:
            instructions, instruction_lengths = self.encode_instruction(
                instruction, device
            )

        return self.model.inference(
            content, task, is_time_aligned, instructions, instruction_lengths,
            num_steps, sway_sampling_coef, guidance_scale, disable_progress
        )
