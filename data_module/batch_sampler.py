import math
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset
from tqdm import trange

from data_module.dataset import TaskGroupedAudioGenConcatDataset
from data_module.sampler import TaskIteratingSampler


class DataSourceGetLengthMixin(Dataset):
    def get_length(self, idx: int) -> float | int:
        raise NotImplementedError("Subclasses must implement this method")


class DynamicBatchSampler(BatchSampler):
    """Base class for dynamic batching based on frame length threshold.
    
    Creates batches dynamically to ensure total frames per batch does not exceed
    the threshold, improving padding efficiency.

    Args:
        data_source (Dataset): The dataset to sample from. 
            Must implement a `get_length(idx)` method returning the length of a data point.
        batch_length_threshold (int): The maximum total length per batch.
        max_samples_per_batch (int, optional): The maximum number of samples per batch, 0 means no limit. Default is 0.
        random_seed (int, optional): Seed for randomization. Default is None.
        drop_last (bool, optional): If True, drops the last batch if it's smaller than the threshold. Default is False.
    """
    def __init__(
        self,
        data_source: DataSourceGetLengthMixin,
        batch_length_threshold: int | float,
        max_samples_per_batch: int = 0,
        random_seed: int = None,
        drop_last: bool = False
    ):
        self.batch_length_threshold = batch_length_threshold
        self.max_samples_per_batch = max_samples_per_batch
        self.random_seed = random_seed
        self.epoch = 0
        self.data_source = data_source

        batches = []
        batch = []
        batch_length = 0

        indices_with_lengths = self.get_indices_with_lengths()

        for idx, sample_length in indices_with_lengths:
            if batch_length + sample_length <= self.batch_length_threshold and (
                max_samples_per_batch == 0 or
                len(batch) < max_samples_per_batch
            ):
                batch.append(idx)
                batch_length += sample_length
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if sample_length <= self.batch_length_threshold:
                    batch = [idx]
                    batch_length = sample_length
                else:
                    batch = []
                    batch_length = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        self.batches = batches
        self.drop_last = drop_last

    def get_indices_with_lengths(self, ) -> list[tuple[int, float | int]]:
        result = []
        for idx in trange(len(self.data_source)):
            result.append((idx, self.data_source.get_length(idx)))
        return result

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


class SortedDynamicBatchSampler(DynamicBatchSampler):
    """Dynamic batch sampler with length-based sorting.
    
    First sorts samples by frame length, then creates dynamic batches.
    This improves padding efficiency by grouping similar-length samples together.
    """
    def get_indices_with_lengths(self) -> list[tuple[int, float | int]]:
        return sorted(
            super().get_indices_with_lengths(), key=lambda elem: elem[1]
        )


class TaskGroupedIteratingBatchSampler(BatchSampler):
    """
    Batch sampler that yields batches whose samples all come from the
    same task. Tasks are visited round-robin: batch1 (task1), batch2 (task2),
    It is *infinite*; stop when the enclosing `DataLoader` has produced enough batches.
    """
    def __init__(
        self,
        data_source: TaskGroupedAudioGenConcatDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive int")

        self.tasks = data_source.tasks  # e.g. ["task1", "task2", ...]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.task_data_ptr = {}
        self.task_data_sizes = {}
        self.task_data_idxs = {}

        for task in self.tasks:
            self.task_data_ptr[task] = 0
            self.task_data_sizes[task] = int(
                data_source.task_to_cum_sum_lengths[task][-1]
            )
            self.task_data_idxs[task] = np.arange(self.task_data_sizes[task])
            if shuffle:
                np.random.shuffle(self.task_data_idxs[task])

    def __iter__(self):
        task_ptr = 0
        num_tasks = len(self.tasks)

        while True:
            task = self.tasks[task_ptr]
            idx_list = self.task_data_idxs[task]
            data_ptr = self.task_data_ptr[task]

            # build a batch with all samples from the same task
            batch = []
            for _ in range(self.batch_size):
                batch.append((task, idx_list[data_ptr]))

                # advance pointer
                data_ptr = (data_ptr + 1) % self.task_data_sizes[task]
                if data_ptr == 0 and self.shuffle:  # epoch over for this task
                    np.random.shuffle(self.task_data_idxs[task])

            # update `task_data_ptr`
            self.task_data_ptr[task] = data_ptr

            yield batch

            # advance to next task
            task_ptr = (task_ptr + 1) % num_tasks

    def __len__(self):
        # unused for an infinite sampler
        return max(self.task_data_sizes.values()) // self.batch_size


class TaskGroupedDynamicBatchSampler(BatchSampler):
    """
    Batch sampler that yields batches whose samples all come from the
    same task. Samples are sorted by length, then grouped into batches.
    """
    def __init__(
        self,
        data_source: TaskGroupedAudioGenConcatDataset,
        batch_length_threshold: int | float,
        max_samples_per_batch: int = 0,
        task_sampling_weights: dict = None,
        random_seed: int | None = None,
        drop_last: bool = False,
        sort_length: bool = True,
        same_task_batch_group_size: int | None = None,
    ):
        self.batch_length_threshold = batch_length_threshold
        self.max_samples_per_batch = max_samples_per_batch
        self.random_seed = random_seed
        self.data_source = data_source
        self.drop_last = drop_last
        self.sort_length = sort_length
        same_task_batch_group_size = 1
        self.same_task_batch_group_size = self.resolve_group_size(
            same_task_batch_group_size
        )

        self.tasks = data_source.tasks
        if task_sampling_weights:
            for key, weight in task_sampling_weights.items():
                if key in self.tasks:
                    self.tasks += [key] * (weight - 1)
        print(f'task sample order: {self.tasks}')
        print(f'same task batch group size: {self.same_task_batch_group_size}')

        self.batches = []
        self.batch = []
        self.batch_length = 0

        self.task_to_batches = {}
        self.task_batch_ptr = {}
        self.task_to_epoch = {}
        for task in self.tasks:
            self.task_to_batches[task] = self.build_batches(task)
            self.task_batch_ptr[task] = 0
            self.task_to_epoch[task] = 0

    def __iter__(self):
        task_ptr = 0
        num_tasks = len(self.tasks)

        while True:
            task = self.tasks[task_ptr]
            for _ in range(self.same_task_batch_group_size):
                yield self.next_task_batch(task)
            task_ptr = (task_ptr + 1) % num_tasks

    def __len__(self):
        return max(len(v) for v in self.task_to_batches.values())

    def resolve_group_size(self, group_size: int | None) -> int:
        if group_size is None:
            if dist.is_available() and dist.is_initialized():
                group_size = dist.get_world_size()
            else:
                group_size = int(os.environ.get("WORLD_SIZE", "1"))
        if group_size <= 0:
            raise ValueError("same_task_batch_group_size must be positive")
        return group_size

    def next_task_batch(self, task: str):
        batch_ptr = self.task_batch_ptr[task]
        batch = self.task_to_batches[task][batch_ptr]

        batch_ptr = (batch_ptr + 1) % len(self.task_to_batches[task])
        if batch_ptr == 0:
            if self.random_seed is not None:
                g = torch.Generator()
                g.manual_seed(self.random_seed + self.task_to_epoch[task])
                indices = torch.randperm(
                    len(self.task_to_batches[task]), generator=g
                ).tolist()
                self.task_to_batches[task] = [
                    self.task_to_batches[task][i] for i in indices
                ]
            self.task_to_epoch[task] = self.task_to_epoch[task] + 1

        self.task_batch_ptr[task] = batch_ptr
        return batch

    def get_updated_batch_length(
        self, max_length_in_batch: int, sample_length: int, batch_size: int
    ) -> int:
        max_length_in_batch = max(max_length_in_batch, sample_length)
        return max_length_in_batch * (1 + batch_size)

    def build_batches(self, task: str):
        batches = []
        batch = []

        indices_with_lengths = []
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed)
            indices = torch.randperm(
                int(self.data_source.task_to_cum_sum_lengths[task][-1]),
                generator=g
            ).tolist()
        else:
            indices = range(
                int(self.data_source.task_to_cum_sum_lengths[task][-1])
            )
        for idx in indices:
            indices_with_lengths.append(
                (idx, self.data_source.get_length((task, idx)))
            )

        if self.sort_length:
            indices_with_lengths = sorted(
                indices_with_lengths, key=lambda elem: elem[1]
            )

        max_length_in_batch = -1
        for idx, sample_length in indices_with_lengths:

            updated_batch_length = self.get_updated_batch_length(
                max_length_in_batch, sample_length, len(batch)
            )

            if updated_batch_length <= self.batch_length_threshold and (
                self.max_samples_per_batch == 0 or  # no batch size upper limit
                len(batch) < self.max_samples_per_batch
            ):
                batch.append((task, idx))
                max_length_in_batch = max(max_length_in_batch, sample_length)
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if sample_length <= self.batch_length_threshold:
                    batch = [(task, idx)]
                    max_length_in_batch = sample_length
                else:
                    batch = []
                    max_length_in_batch = -1

        if not self.drop_last and len(batch) > 0:
            batches.append(batch)

        if self.random_seed is not None:
            indices = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in indices]

        return batches


class TaskIteratingDynamicBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: TaskIteratingSampler,
        batch_length_threshold: int | float,
        max_samples_per_batch: int = 0,
    ):
        self.sampler = sampler
        self.batch_length_threshold = batch_length_threshold
        self.max_samples_per_batch = max_samples_per_batch
        self.drop_last = True  # required by accel_hydra

    def get_updated_batch_length(
        self, max_length_in_batch: int, sample_length: int, batch_size: int
    ) -> int:
        max_length_in_batch = max(max_length_in_batch, sample_length)
        return max_length_in_batch * (1 + batch_size)

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        max_length_in_batch = -1
        batch = []
        while True:
            task, idx = next(sampler_iter)
            sample_length = self.sampler.data_source.get_length((task, idx))
            updated_batch_length = self.get_updated_batch_length(
                max_length_in_batch, sample_length, len(batch)
            )
            if updated_batch_length <= self.batch_length_threshold and (
                self.max_samples_per_batch == 0 or
                len(batch) < self.max_samples_per_batch
            ):
                batch.append((task, idx))
                max_length_in_batch = max(max_length_in_batch, sample_length)
            else:
                if len(batch) > 0:
                    yield batch
                if sample_length <= self.batch_length_threshold:
                    batch = [(task, idx)]
                    max_length_in_batch = sample_length
                else:
                    batch = []
                    max_length_in_batch = -1

    def __len__(self):
        # unused for infinite batch sampler
        return sum(
            self.sampler.task_data_sizes.values()
        ) // self.max_samples_per_batch


class TaskGroupedSequentialBatchSampler(BatchSampler):
    """
    Batch sampler that yields batches whose samples all come from the
    same task. 
    """
    def __init__(
        self,
        data_source: TaskGroupedAudioGenConcatDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive int")

        self.tasks: list[str] = list(data_source.tasks)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.task_data_idxs: dict[str, np.ndarray] = {}
        self.task_data_sizes: dict[str, int] = {}
        self.task_data_ptr: dict[str, int] = {}

        for task in self.tasks:
            size = int(data_source.task_to_cum_sum_lengths[task][-1])
            self.task_data_sizes[task] = size
            self.task_data_ptr[task] = 0

            idxs = np.arange(size, dtype=np.int64)
            if shuffle:
                np.random.shuffle(idxs)
            self.task_data_idxs[task] = idxs

        self._num_batches = 0
        for size in self.task_data_sizes.values():
            if drop_last:
                self._num_batches += size // batch_size
            else:
                self._num_batches += math.ceil(size / batch_size)

    def __iter__(self):
        for task in self.tasks:
            idxs = self.task_data_idxs[task]
            size = self.task_data_sizes[task]
            ptr = 0

            while ptr < size:
                end = min(ptr + self.batch_size, size)
                batch_idxs = idxs[ptr:end]
                if len(batch_idxs) < self.batch_size and self.drop_last:
                    break

                yield [(task, idx) for idx in batch_idxs]
                ptr = end

    def __len__(self) -> int:
        return self._num_batches
