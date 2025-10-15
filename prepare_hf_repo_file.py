from pathlib import Path

import fire
from omegaconf import OmegaConf


def main(
    exp_config: str = "",
    # ckpt_path: str = "",
    target_dir: str = "",
):

    exp_config = Path(exp_config)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(exp_config / "config.yaml")

    config["model"]["autoencoder"]["pretrained_ckpt"] = (
        Path("vae") /
        Path(config["model"]["autoencoder"]["pretrained_ckpt"]).name
    ).__str__()

    OmegaConf.save({
        "sample_rate": config["sample_rate"],
        "model": config["model"]
    }, target_dir / "config.yaml")


if __name__ == "__main__":
    fire.Fire(main)
