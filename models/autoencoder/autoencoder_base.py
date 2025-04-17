from abc import abstractmethod, ABC
from typing import Sequence
import torch
import torch.nn as nn


class AutoEncoderBase(ABC):
    def __init__(
        self, downsampling_ratio: int, sample_rate: int,
        latent_shape: Sequence[int | None]
    ):
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_token_rate = sample_rate // downsampling_ratio
        self.latent_shape = latent_shape
        self.time_dim = latent_shape.index(None) + 1  # the first dim is batch

    @abstractmethod
    def encode(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
