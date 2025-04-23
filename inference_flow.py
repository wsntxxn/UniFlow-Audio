from pathlib import Path

import soundfile as sf
import torch
import hydra
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from tqdm import tqdm

from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip

from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()

def merge_audio_video(waveform_path, audio_id, target_dir, sr):
    video_id = audio_id
    video_path = Path("/hpc_stor03/public/shared/data/raa/VGGSound/data") / f"{video_id}"
    video = VideoFileClip(str(video_path))
    print(f"Video duration: {video.duration} seconds")
    duration = video.duration
    
    audio = AudioFileClip(waveform_path)
    audio_clip = audio.subclip(0, min(audio.duration, video.duration))

    video = video.set_audio(audio_clip)
    video.write_videofile(f'{target_dir}')

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = []

    @hydra.main(config_path="configs", config_name="inference_flow")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    if "exp_dir" in config:
        exp_dir = Path(config["exp_dir"])
        ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                )[-1] / "model.safetensors"
    elif "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir.parent.parent

    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)

    model = model.to(device)
    test_dataloader = hydra.utils.instantiate(
        config["test_dataloader"], _convert_="all"
    )
    model.eval()
    
    audio_output_dir = exp_dir / config["wav_dir"]
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            for key in list(batch.keys()):
                data = batch[key]
                if isinstance(data, torch.Tensor):
                    batch[key] = data.to(device)

            waveform = model.inference(
                # latent_shape=config["latent_shape"],
                num_inference_steps=config["num_steps"],
                guidance_scale=config["guidance_scale"],
                use_gt_duration=config["use_gt_duration"],
                **batch
            )
            for i in range(len(waveform)):
                if isinstance(batch["content"][i], str):
                    out_file: str = batch["content"][i]
                else:
                    out_file: str = batch["audio_id"][i]
                if not out_file.endswith(".wav"):
                    out_file = f"{out_file}.wav"

                task = batch['task'][i]
                if task == "video_to_audio":
                    audio_id = str(batch["audio_id"][i])
                    target_dir = audio_output_dir / f'{batch["label"][i][0].decode()}_{audio_id}'
                    wav_path = f'{str(target_dir)}.wav'
                elif task == "text_to_audio":
                    audio_id = str(batch["audio_id"][i])
                    target_dir = audio_output_dir / f'{batch["content"][i]}'
                    wav_path = f'{str(target_dir)}.wav'

                sf.write(
                    wav_path,
                    waveform[i, 0].cpu().numpy(),
                    samplerate=exp_config["sample_rate"],
                )
                if task == "video_to_audio":
                    merge_audio_video(wav_path, audio_id, str(target_dir), sr=exp_config["sample_rate"])




if __name__ == "__main__":
    main()
