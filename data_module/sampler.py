import math
import numpy as np
from torch.utils.data import Sampler, BatchSampler

from data_module.dataset import TaskGroupedAudioGenConcatDataset


class TaskIteratingSampler(Sampler):

    def __init__(self,
                 data_source: TaskGroupedAudioGenConcatDataset,
                 shuffle: bool = True,
                 task_sampling_weights: dict = None):
        self.tasks = data_source.tasks
        # add extra task sampling times for some tasks
        if task_sampling_weights:
            for key, weight in task_sampling_weights.items():
                if key in self.tasks:
                    self.tasks += [key] * (weight - 1)
        print(f'task sample order:{self.tasks}')
        # pointers & indices for each task

        # pointer to the data index of each task, iterating like 0, 1, 2, ...
        self.task_data_ptr = {}
        # total data size for each task
        self.task_data_sizes = {}
        # (shuffled) dataset index list for each task, can be used with `task`
        # to retrieve item from `data_source`: data_source[(task, data_idx)]
        self.task_data_idxs = {}
        for task in self.tasks:
            self.task_data_ptr[task] = 0
            self.task_data_sizes[task] = int(
                data_source.task_to_cum_sum_lengths[task][-1])
            self.task_data_idxs[task] = np.arange(self.task_data_sizes[task])
            if shuffle:
                np.random.shuffle(self.task_data_idxs[task])

        self.shuffle = shuffle

    def __iter__(self):
        task_ptr = 0
        num_tasks = len(self.tasks)
        while True:
            task = self.tasks[task_ptr]
            idx_list = self.task_data_idxs[task]
            data_ptr = self.task_data_ptr[task]
            yield task, idx_list[data_ptr]
            # advance pointer
            self.task_data_ptr[task] = (data_ptr +
                                        1) % self.task_data_sizes[task]
            if self.task_data_ptr[task] == 0 and self.shuffle:
                np.random.shuffle(self.task_data_idxs[task])
            # advance to next task
            task_ptr = (task_ptr + 1) % num_tasks

    def __len__(self):
        # unused for an infinite sampler
        return max(self.task_data_sizes.values())


class InferenceTaskIteratingSampler(Sampler):
    # Finite sampler for inference
    def __init__(self,
                 data_source,
                 shuffle=False,
                 ):
        self.tasks = data_source.tasks
        self.task_data_idxs = {}
        for task in self.tasks:
            num_samples = int(data_source.task_to_cum_sum_lengths[task][-1])
            self.task_data_idxs[task] = list(np.arange(num_samples))
            if shuffle:
                np.random.shuffle(self.task_data_idxs[task])
        self.active_tasks = list(self.tasks)  # tasks still having data
        self.task_ptr = 0

    def __iter__(self):
        while self.active_tasks:
            task = self.active_tasks[self.task_ptr]
            if self.task_data_idxs[task]:
                idx = self.task_data_idxs[task].pop(0)
                yield task, idx
            # remove exhausted tasks
            if not self.task_data_idxs[task]:
                self.active_tasks.pop(self.task_ptr)
                if not self.active_tasks:
                    break
                self.task_ptr = self.task_ptr % len(self.active_tasks)
            else:
                self.task_ptr = (self.task_ptr + 1) % len(self.active_tasks)

    def __len__(self):
        return sum(len(v) for v in self.task_data_idxs.values())


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
                data_source.task_to_cum_sum_lengths[task][-1])
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
        for task in self.tasks:  # 顺序遍历 task
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
