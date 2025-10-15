from pathlib import Path
import os

import soundfile as sf
import torch
import hydra
from accelerate import Accelerator

from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from tqdm import tqdm

from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase
from utils.general import sanitize_filename

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()


def main():

    accelerator = Accelerator(mixed_precision="no")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    exp_dir, ckpt_dir = None, None
    if os.path.isdir(config["exp_dir"]):
        exp_dir = Path(config["exp_dir"])
    if os.path.isdir(config["ckpt_dir_or_file"]):
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
    elif os.path.isfile(config["ckpt_dir_or_file"]):
        ckpt_path = config["ckpt_dir_or_file"]
        ckpt_dir = Path(ckpt_path).parent

    if ckpt_dir is None and exp_dir is None:
        if not os.path.exists(config["ckpt_dir"]):
            raise ValueError(f"ckpt_dir {config['ckpt_dir']} does not exist.")
        raise ValueError("Either exp_dir or ckpt_dir should be provided.")

    if ckpt_dir is None:
        use_best = config.get("use_best", True)
        if use_best:  # use best ckpt
            ckpt_path: Path = exp_dir / "checkpoints/best/model.safetensors"
        else:  # use last ckpt
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                    )[-1] / "model.safetensors"
    if exp_dir is None:
        exp_dir = ckpt_dir.parent.parent

    if "exp_config_path" in config:
        exp_config_path = config['exp_config_path']
        exp_config = OmegaConf.load(config['exp_config_path'])
    else:
        exp_config_path = exp_dir / "config.yaml"

    exp_config = OmegaConf.load(exp_config_path)
    accelerator.print(
        f'\n ckpt path: {ckpt_path}, model config path: {exp_config_path}\n '
    )

    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)

    if "sampler" in config["test_dataloader"]:
        data_source = hydra.utils.instantiate(
            config["test_dataloader"]["dataset"], _convert_="all"
        )
        sampler = hydra.utils.instantiate(
            config["test_dataloader"]["sampler"],
            data_source=data_source,
            _convert_="all"
        )
        test_dataloader = hydra.utils.instantiate(
            config["test_dataloader"], sampler=sampler, _convert_="all"
        )
    else:
        test_dataloader = hydra.utils.instantiate(
            config["test_dataloader"], _convert_="all"
        )

    model.eval()

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    if "noise_scheduler" in config:
        scheduler = getattr(
            noise_schedulers,
            config["noise_scheduler"]["type"],
        ).from_pretrained(
            config["noise_scheduler"]["name"],
            subfolder="scheduler",
        )
    else:
        scheduler = None

    if config["wav_dir_root"] is not None:
        wav_dir_root = Path(config["wav_dir_root"])
    else:
        wav_dir_root = exp_dir
    audio_output_dir = wav_dir_root / config["wav_dir"]
    if accelerator.is_main_process:
        audio_output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    pbar_disable = not accelerator.is_main_process

    with torch.no_grad():

        for batch in tqdm(test_dataloader, disable=pbar_disable):
            kwargs = config["infer_args"].copy()
            kwargs.update(batch)

            waveform = unwrapped_model.inference(
                scheduler=scheduler,
                **kwargs,
            )

            for name, wave, task in zip(
                batch["item_name"], waveform, batch["task"]
            ):
                (audio_output_dir / task).mkdir(parents=True, exist_ok=True)
                safe_name = sanitize_filename(name)
                sf.write(
                    audio_output_dir / task / f"{safe_name}.wav",
                    wave[0].cpu().numpy(),
                    samplerate=exp_config["sample_rate"],
                )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
