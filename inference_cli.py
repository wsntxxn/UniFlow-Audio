#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Callable

import fire
import torch
import torchaudio
import soundfile as sf
import numpy as np

from modeling_uniflow_audio import UniFlowAudioModel
from constants import TIME_ALIGNED_TASKS, NON_TIME_ALIGNED_TASKS
from utils.phonemize import sentence_to_phones


class InferenceCLI:
    def __init__(self):
        self.model_name_or_path = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.g2p = None
        self.word2phone = None
        self.speaker_model = None
        self.svs_processor = None
        self.singer_mapping = None

        self.video_preprocessor = None
        self.video_size = (256, 256)
        self.video_fps = 10

    def init_model(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        if Path(self.model_name_or_path).exists():
            self.model = UniFlowAudioModel(model_name_or_path)
        else:
            self.model = UniFlowAudioModel(f"wsntxxn/{model_name_or_path}")
        self.model.to(self.device)
        self.sample_rate = self.model.config["sample_rate"]

    def init_speaker_model(self, ):
        import wespeaker

        if self.speaker_model is None:
            self.speaker_model = wespeaker.load_model("english")
            self.speaker_model.set_device(self.device)

    def init_svs_processor(self, ):
        from utils.diffsinger_utilities import SVSInputConverter, TokenTextEncoder

        if self.svs_processor is None:
            phoneme_list = json.load(open(self.model.svs_phone_set_path, "r"))
            self.svs_processor = {
                "converter":
                    SVSInputConverter(
                        self.model.svs_singer_mapping, self.model.svs_pinyin2ph
                    ),
                "tokenizer":
                    TokenTextEncoder(
                        None, vocab_list=phoneme_list, replace_oov=','
                    )
            }

    def init_video_preprocessor(self, ):
        if self.video_preprocessor is None:
            from third_party.mmaudio.feature_extractor import init_feature_extractor
            self.video_preprocessor = init_feature_extractor()

    def on_inference_start(self, model_name_or_path):
        if self.model_name_or_path is None or model_name_or_path != self.model_name_or_path:
            self.init_model(model_name_or_path)

    @staticmethod
    def add_prehook(func: Callable, ):
        def wrapper(self, *args, **kwargs):
            model_name_or_path = kwargs.get(
                "model_name_or_path", "UniFlow-Audio-large"
            )
            self.on_inference_start(model_name_or_path)
            return func(self, *args, **kwargs)

        return wrapper

    @add_prehook
    def t2a(
        self,
        caption: str,
        model_name_or_path: str = "UniFlow-Audio-large",
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        self._run_inference(
            content=caption,
            task="text_to_audio",
            instruction=instruction,
            instruction_idx=instruction_idx,
            model_name_or_path=model_name_or_path,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )

    @add_prehook
    def t2m(
        self,
        caption: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        self._run_inference(
            content=caption,
            task="text_to_music",
            model_name_or_path=model_name_or_path,
            instruction=instruction,
            instruction_idx=instruction_idx,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path,
        )

    @add_prehook
    def tts(
        self,
        transcript: str,
        ref_speaker_speech: str,
        model_name_or_path: str = "UniFlow-Audio-large",
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):

        from montreal_forced_aligner.g2p.generator import PyniniConsoleGenerator

        self.init_speaker_model()

        if not self.g2p:
            self.g2p = PyniniConsoleGenerator(
                g2p_model_path=self.model.g2p_model_path,
                strict_graphemes=False,
                num_pronunciations=1,
                include_bracketed=False
            )
            self.g2p.setup()

        if not self.word2phone:
            self.word2phone = json.load(
                open(
                    self.model.tts_word2phone_dict_path, "r", encoding="utf-8"
                )
            )

        # OOV word will use a g2p model to predict phoneme
        phonemes, OOV_list = sentence_to_phones(
            transcript, self.word2phone, self.g2p
        )

        print(phonemes)
        phone_indices = [
            self.model.tts_phone2id.get(
                p, self.model.tts_phone2id.get("spn", 0)
            ) for p in phonemes
        ]

        prompt_waveform, orig_sr = torchaudio.load(ref_speaker_speech)
        prompt_waveform = prompt_waveform.mean(0)
        prompt_waveform = torchaudio.functional.resample(
            prompt_waveform,
            orig_freq=orig_sr,
            new_freq=self.model.tts_prompt_sr
        )
        content = {
            "phoneme": np.array(phone_indices, dtype=np.int64),
            "prompt_waveform": prompt_waveform,
        }
        self._run_inference(
            content=content,
            task="text_to_speech",
            model_name_or_path=model_name_or_path,
            instruction=instruction,
            instruction_idx=instruction_idx,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path,
        )

    @add_prehook
    def _audio_input_inference(
        self,
        input_audio: str,
        task: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        waveform, orig_sr = torchaudio.load(input_audio)
        waveform = waveform.mean(0)
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=orig_sr, new_freq=self.sample_rate
        )
        self._run_inference(
            content=waveform,
            task=task,
            instruction=instruction,
            instruction_idx=instruction_idx,
            model_name_or_path=model_name_or_path,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )

    def se(
        self,
        noisy_speech: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 1.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        self._audio_input_inference(
            input_audio=noisy_speech,
            task="speech_enhancement",
            instruction=instruction,
            instruction_idx=instruction_idx,
            model_name_or_path=model_name_or_path,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )

    def sr(
        self,
        low_sr_audio: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 1.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        self._audio_input_inference(
            input_audio=low_sr_audio,
            task="audio_super_resolution",
            instruction=instruction,
            instruction_idx=instruction_idx,
            model_name_or_path=model_name_or_path,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )

    @add_prehook
    def v2a(
        self,
        video: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.mp4",
    ):
        from utils.video import merge_audio_video
        from third_party.mmaudio.feature_extractor import get_video_feature

        self.init_video_preprocessor()
        video_feature = get_video_feature(video, self.video_preprocessor)

        waveform = self._run_inference(
            content=video_feature,
            task="video_to_audio",
            model_name_or_path=model_name_or_path,
            instruction=instruction,
            instruction_idx=instruction_idx,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path,
        )

        merge_audio_video(
            waveform, video, output_path, audio_fps=self.sample_rate
        )

    @add_prehook
    def svs(
        self,
        singer: str,
        music_score: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        self.init_svs_processor()
        text, note, note_dur = music_score.split('<sep>')
        if singer not in self.model.svs_singer_mapping:
            print(f"Unsupported singer {singer}, available singers: ")
            print(list(self.model.svs_singer_mapping.keys()))
            raise KeyError

        midi = self.svs_processor["converter"].preprocess_input({
            "spk_name": singer,
            "text": text,
            "notes": note,
            "notes_duration": note_dur,
        })
        midi["phoneme"] = self.svs_processor["tokenizer"].encode(
            midi["phoneme"]
        )
        self._run_inference(
            content=midi,
            task="singing_voice_synthesis",
            model_name_or_path=model_name_or_path,
            instruction=instruction,
            instruction_idx=instruction_idx,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path,
        )

    def _run_inference(
        self,
        content: Any,
        task: str,
        model_name_or_path: str,
        instruction: str | None = None,
        instruction_idx: int | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 25,
        output_path: str = "./output.wav",
    ):
        if self.model_name_or_path is None or model_name_or_path != self.model_name_or_path:
            self.init_model(model_name_or_path)
        time_aligned = torch.zeros(1, 2).bool()
        if task in TIME_ALIGNED_TASKS:
            time_aligned[0, 0] = True
        if task in NON_TIME_ALIGNED_TASKS:
            time_aligned[0, 1] = True
        if instruction:
            instruction = [instruction]
        if instruction_idx:
            instruction_idx = [instruction_idx]

        waveform = self.model.sample(
            content=[content],
            task=[task],
            time_aligned=time_aligned,
            instruction=instruction,
            instruction_idx=instruction_idx,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            disable_progress=False
        )
        waveform = waveform[0, 0].cpu().numpy()

        output_path = output_path.__str__()
        if not output_path.endswith(".mp4"):
            sf.write(output_path, waveform, self.sample_rate)

        return waveform


if __name__ == "__main__":
    fire.Fire(InferenceCLI)
