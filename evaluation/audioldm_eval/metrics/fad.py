"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import torch
from pathlib import Path

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from scipy import linalg
from tqdm import tqdm
from audioldm_eval.datasets.load_mel import WaveDataset
from torch.utils.data import DataLoader


class FrechetAudioDistance:
    def __init__(
        self,
        use_pca=False,
        use_activation=False,
        verbose=False,
        device="cuda",
        audio_load_worker=8
    ):
        self.device = device
        self.verbose = verbose
        if dist.is_initialized():
            if dist.get_rank() == 0:
                self.disable_progress = not self.verbose
            else:
                self.disable_progress = True
        else:
            self.disable_progress = not self.verbose
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.audio_load_worker = audio_load_worker

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        torch_home_dir = os.environ.get(
            "TORCH_HOME", os.path.expanduser("~/.cache/torch")
        )
        ckpt_path = os.path.join(
            torch_home_dir, "hub/harritaylor_torchvggish_master"
        )
        if os.path.exists(ckpt_path):
            self.model = torch.hub.load(ckpt_path, "vggish", source="local")
        else:
            self.model = torch.hub.load("harritaylor/torchvggish", "vggish")

        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()
        self.model.to(self.device)
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.device(self.device).type == "cuda":
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[local_rank]
                )
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model
                )

    def load_audio_data(self, x):
        dataset = WaveDataset(x, 16000, limit_num=None)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False
            )
        else:
            sampler = None
        outputloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=4,
            drop_last=False
        )
        data_list = []
        for batch in tqdm(
            outputloader,
            disable=self.disable_progress,
            desc="Loading data to RAM"
        ):
            data_list.append((batch[0][0, 0], 16000))
        return data_list

    def get_embeddings(self, x: str | dict, sr=16000, limit_num=None):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        x = self.load_audio_data(x)
        if isinstance(x, list):
            try:
                with torch.no_grad():
                    for audio, sr in tqdm(
                        x,
                        disable=self.disable_progress,
                        desc="Calculating VGGish embeddings"
                    ):
                        embd = self.model.forward(audio.numpy(), sr)
                        embd = embd.cpu().numpy()
                        embd_lst.append(embd)
            except Exception as e:
                print(
                    "[Frechet Audio Distance] get_embeddings throw an exception: {}"
                    .format(str(e))
                )
        else:
            raise AttributeError

        embds = np.concatenate(embd_lst, axis=0)
        if dist.is_available() and dist.is_initialized():
            gathered_embds = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_embds, embds)
            embds = np.concatenate(gathered_embds, axis=0)
        return embds

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1
                                        ) + np.trace(sigma2) - 2 * tr_covmean

    def score(
        self,
        background_files: str | dict,
        eval_files: str | dict,
        store_embds=False,
        limit_num=None,
        recalculate=False
    ):
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        try:
            is_main_rank = (dist.is_initialized() and
                            dist.get_rank() == 0) or not dist.is_initialized()
            if isinstance(background_files, str):
                fad_generated_folder_cache = background_files + "_fad_feature_cache.npy"
            elif isinstance(background_files, dict):
                first_file = next(iter(background_files.values()))
                background_dir = Path(first_file).parent
                fad_generated_folder_cache = background_dir.with_name(
                    background_dir.stem + "_fad_feature_cache.npy"
                )

            if (not os.path.exists(fad_generated_folder_cache) or recalculate):
                embds_background = self.get_embeddings(
                    background_files, limit_num=limit_num
                )
                if is_main_rank:
                    np.save(fad_generated_folder_cache, embds_background)
            else:
                if is_main_rank:
                    print(
                        "Reload fad_generated_folder_cache",
                        fad_generated_folder_cache
                    )
                embds_background = np.load(fad_generated_folder_cache)

            if isinstance(eval_files, str):
                fad_target_folder_cache = eval_files + "_fad_feature_cache.npy"
            elif isinstance(eval_files, dict):
                first_file = next(iter(eval_files.values()))
                eval_dir = Path(first_file).parent
                fad_target_folder_cache = eval_dir.with_name(
                    eval_dir.stem + "_fad_feature_cache.npy"
                )

            if (not os.path.exists(fad_target_folder_cache) or recalculate):
                embds_eval = self.get_embeddings(
                    eval_files, limit_num=limit_num
                )
                if is_main_rank:
                    np.save(fad_target_folder_cache, embds_eval)
            else:
                if is_main_rank:
                    print(
                        "Reload fad_target_folder_cache",
                        fad_target_folder_cache
                    )
                embds_eval = np.load(fad_target_folder_cache)

            if store_embds and is_main_rank:
                np.save("embds_background.npy", embds_background)
                np.save("embds_eval.npy", embds_eval)

            if len(embds_background) == 0:
                if is_main_rank:
                    print(
                        "[Frechet Audio Distance] background set dir is empty, exitting..."
                    )
                return -1

            if len(embds_eval) == 0:
                if is_main_rank:
                    print(
                        "[Frechet Audio Distance] eval set dir is empty, exitting..."
                    )
                return -1

            mu_background, sigma_background = self.calculate_embd_statistics(
                embds_background
            )
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, sigma_background, mu_eval, sigma_eval
            )

            return {"frechet_audio_distance": fad_score}

        except Exception as e:
            if is_main_rank:
                print(
                    "[Frechet Audio Distance] exception thrown, {}".format(
                        str(e)
                    )
                )
            return -1
