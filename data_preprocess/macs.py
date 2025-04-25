from pathlib import Path
import pandas as pd
import json
import tqdm
import yaml
import os
PREPARED_AUDIOCAPS_DIR = Path(
    "/hpc_stor03/sjtu_home/zeyu.xie/maa/x2audio_data/macs/raw_data"
)
TARGET_AUDIOCAPS_DIR = Path("/hpc_stor03/sjtu_home/zeyu.xie/maa/x2audio_data/macs")
ORI_DATA_DIR = Path("/hpc_stor03/public/shared/data/raa/DCASE2019/task1/task1a/TAU-urban-acoustic-scenes-2019-development/audio")

for split in ["dev"]:
    with open(PREPARED_AUDIOCAPS_DIR / "MACS.yaml", "r") as yaml_file:
        annotation_yaml = yaml.safe_load(yaml_file)  

        (TARGET_AUDIOCAPS_DIR / "jsonl" / split).mkdir(parents=True, exist_ok=True)
        with open(TARGET_AUDIOCAPS_DIR / "jsonl" / split / "audio.jsonl", "w") as audio_writer, \
            open(TARGET_AUDIOCAPS_DIR / "jsonl" / split / "caption.jsonl", "w") as caption_writer:
            for item in tqdm.tqdm(annotation_yaml["files"]):
                tmp_ori_path = (ORI_DATA_DIR / item['filename'])
                tmp_tgt_path = (TARGET_AUDIOCAPS_DIR / "audio" / item['filename'])
                cmd = f"ln -s {tmp_ori_path} {tmp_tgt_path}"
                os.system(cmd)
                if Path(tmp_ori_path).exists():
                    audio_writer.write(
                        json.dumps({
                            "audio_id": item["filename"],
                            "audio": f"macs/audio/{item['filename']}"
                        }) + "\n"
                    )
                else:
                    print(tmp_ori_path)

                for caption_item in item["annotations"]: 
                    caption_writer.write(
                        json.dumps({
                            "audio_id": item["filename"],
                            "caption": caption_item["sentence"]
                        }) + "\n"
                    )
