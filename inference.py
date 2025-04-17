from pathlib import Path

import soundfile as sf
import torch
import hydra
import json
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from tqdm import tqdm

from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase

register_omegaconf_resolvers()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]
    print(f'config: {json.dumps(config, indent=4)}')

    if "exp_dir" in config:
        use_best=config["use_best"]

        exp_dir = Path(config["exp_dir"])
        
        #使用best ckpt
        if use_best:
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                    )[0] / "model.safetensors"
        else:
            # 使用last ckpt
            ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                    )[-1] / "model.safetensors"
    elif "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir.parent.parent
    print(f'ckpt path:{ckpt_path}')
    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    # print(f'model config:{json.dumps(dict(exp_config["model"], indent=4))}')
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)

    model = model.to(device)
    test_dataloader = hydra.utils.instantiate(
        config["test_dataloader"], _convert_="all"
    )
    model.eval()

    scheduler = getattr(
        noise_schedulers,
        config["noise_scheduler"]["type"],
    ).from_pretrained(
        config["noise_scheduler"]["name"],
        subfolder="scheduler",
    )

    audio_output_dir = exp_dir / config["wav_dir"]
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            for key in list(batch.keys()):
                data = batch[key]
                if isinstance(data, torch.Tensor):
                    batch[key] = data.to(device)

            waveform = model.inference(
                scheduler=scheduler,
                latent_shape=config["latent_shape"],
                num_steps=config["num_steps"],
                guidance_scale=config["guidance_scale"],
                use_gt_duration=config["use_gt_duration"],
                **batch
            )

            if isinstance(batch["content"][0], str):
                out_file: str = batch["content"][0]
            else:
                out_file: str = batch["audio_id"][0]
            if not out_file.endswith(".wav"):
                out_file = f"{out_file}.wav"

            sf.write(
                audio_output_dir / out_file,
                waveform[0, 0].cpu().numpy(),
                samplerate=exp_config["sample_rate"],
            )


if __name__ == "__main__":
    main()
