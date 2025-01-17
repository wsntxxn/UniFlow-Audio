from pathlib import Path

import soundfile as sf
import torch
import hydra
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

    exp_dir = Path(config["exp_dir"])
    exp_config = OmegaConf.load(exp_dir / "config.yaml")

    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                            )[-1] / "model.safetensors"
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

    audio_output_dir = exp_dir / "inference"
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
                **batch
            )

            sf.write(
                audio_output_dir / f'{batch["content"][0]}.wav',
                waveform[0, 0].cpu().numpy(),
                samplerate=exp_config["sample_rate"],
            )


if __name__ == "__main__":
    main()
