"""
processes audio metadata and removes entries with NaN values from the corresponding content.jsonl file.

"""
import multiprocessing as mp
import os
import shutil

mp.set_start_method("spawn", force=True)

import hydra
import torch
import json
from omegaconf import OmegaConf
from accelerate import Accelerator
from utils.config import register_omegaconf_resolvers
from models.common import CountParamsBase
from trainer import Trainer
from utils.torch_utilities import check_nan_in_batch
from tqdm import tqdm

register_omegaconf_resolvers()


def main():

    configs = []

    @hydra.main(
        version_base=None, config_path="../configs", config_name="train"
    )
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    # check train split
    collate_fn = hydra.utils.instantiate(
        config["train_dataloader"]['collate_fn'], _convert_="all"
    )
    num_workers = min(mp.cpu_count(), 20)
    data_names = []
    for dataset_config in config["train_dataloader"]['dataset']['datasets']:
        dataset_name = dataset_config['content'].split('/')[-3]
        data_names.append(dataset_name)
    print(f'checking train data integrity in  {data_names}')

    for dataset_config in config["train_dataloader"]['dataset']['datasets']:
        dataset = hydra.utils.instantiate(dataset_config, _convert_="all")
        dataset_name = dataset_config['content'].split('/')[-3]
        print(f'statrt checking train data integrity in  {dataset_name}')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=32,
            num_workers=num_workers
        )
        nan_data_ids = set()
        for batch in tqdm(dataloader):
            nan_ids = check_nan_in_batch(batch)
            if nan_ids:
                print(f'Found {nan_ids} NaN dataline in {dataset_name}')
                nan_data_ids.update(nan_ids)
        content_path = dataset_config['content']

        print(
            f'found  {len(nan_data_ids)} Nan dataline in {dataset_name} train split'
        )
        if len(nan_data_ids) > 0:
            print(
                f'removing {len(nan_data_ids)} Nan dataline in {dataset_name} train split'
            )
            with open(content_path, 'r') as f:
                src_content = f.readlines()
                src_content = [json.loads(line) for line in src_content]
            new_content_data = []
            for line in src_content:
                id_col = line.get('audio_id', None) or line.get('UUID', None)
                if id_col is None:
                    raise ValueError('No audio_id or UUID in content data')
                if id_col in nan_data_ids:
                    continue
                else:
                    new_content_data.append(line)

            bak_content_path = content_path.replace('.jsonl', '_bak.jsonl')
            if os.path.exists(bak_content_path):
                os.remove(bak_content_path)
            shutil.move(content_path, bak_content_path)
            print(
                f'saving {dataset_name} train split origin content data to {bak_content_path}'
            )
            with open(content_path, 'w') as f:
                for line in new_content_data:
                    f.write(json.dumps(line) + '\n')
            print(
                f'saving {dataset_name} train split new content data to {content_path}'
            )

    # check val split
    collate_fn = hydra.utils.instantiate(
        config["val_dataloader"]['collate_fn'], _convert_="all"
    )

    num_workers = min(mp.cpu_count(), 20)

    data_names = []
    for dataset_config in config["val_dataloader"]['dataset']['datasets']:
        dataset_name = dataset_config['content'].split('/')[-3]
        data_names.append(dataset_name)
    print(f'Checking validation data integrity in {data_names}')

    for dataset_config in config["val_dataloader"]['dataset']['datasets']:
        dataset = hydra.utils.instantiate(dataset_config, _convert_="all")
        dataset_name = dataset_config['content'].split('/')[-3]
        print(f'Start checking validation data integrity in {dataset_name}')

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=32,
            num_workers=num_workers
        )

        nan_data_ids = set()

        for batch in tqdm(dataloader):
            nan_ids = check_nan_in_batch(batch)
            if nan_ids:
                print(f'Found {nan_ids} NaN dataline in {dataset_name}')
                nan_data_ids.update(nan_ids)
        print(
            f'found  {len(nan_data_ids)} Nan dataline in {dataset_name} val split'
        )
        if len(nan_data_ids) > 0:
            print(
                f'removing {len(nan_data_ids)} Nan dataline in {dataset_name} val split'
            )

            content_path = dataset_config['content']

            with open(content_path, 'r') as f:
                src_content = f.readlines()
                src_content = [json.loads(line) for line in src_content]

            new_content_data = []
            for line in src_content:
                id_col = line.get('audio_id', None) or line.get('UUID', None)
                if id_col is None:
                    raise ValueError('No audio_id or UUID in content data')
                if id_col in nan_data_ids:
                    continue
                else:
                    new_content_data.append(line)
            print(
                f'Found {len(nan_data_ids)} NaN dataline in {dataset_name} validation split'
            )

            bak_content_path = content_path.replace('.jsonl', '_bak.jsonl')
            if os.path.exists(bak_content_path):
                os.remove(bak_content_path)
            shutil.move(content_path, bak_content_path)
            print(
                f'Saving {dataset_name} validation split original content data to {bak_content_path}'
            )

            with open(content_path, 'w') as f:
                for line in new_content_data:
                    f.write(json.dumps(line) + '\n')
            print(
                f'Saving {dataset_name} validation split new content data to {content_path}'
            )

    print('all done')


if __name__ == "__main__":

    main()
