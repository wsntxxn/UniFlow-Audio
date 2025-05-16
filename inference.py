from pathlib import Path

import soundfile as sf
import torch
import hydra
import re
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from tqdm import tqdm

from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()



MAX_FILE_NAME_LENGTH=50
def sanitize_filename(name, max_len=MAX_FILE_NAME_LENGTH):
    #Clean and truncate a string to make it a valid and safe filename.
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.replace('/', '_')  
    max_len=min(len(name),max_len)
    return name[:max_len]

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    if "exp_dir" in config:
        use_best = config.get("use_best", True)
        exp_dir = Path(config["exp_dir"])
        #use best ckpt
        if use_best:
            ckpt_path: Path = sorted(
                (exp_dir / "checkpoints").iterdir())[0] / "model.safetensors"
        else:
            # use last ckpt
            ckpt_path: Path = sorted(
                (exp_dir / "checkpoints").iterdir())[-1] / "model.safetensors"
    elif "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir.parent.parent
    print(f'\n ckpt path:{ckpt_path}\n ')
    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)

    model = model.to(device)
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

            kwargs = config["infer_args"].copy()
            kwargs.update(batch)
            waveform = model.inference(
                scheduler=scheduler,
                **kwargs,
            )

            for name, wave, task in zip(batch["item_name"], waveform, batch["task"]):
                (audio_output_dir / task).mkdir(parents=True, exist_ok=True)
                safe_name = sanitize_filename(name)
                sf.write(
                    audio_output_dir / task / f"{safe_name}.wav",
                    wave[0].cpu().numpy(),
                    samplerate=exp_config["sample_rate"],
                )


if __name__ == "__main__":
    main()
