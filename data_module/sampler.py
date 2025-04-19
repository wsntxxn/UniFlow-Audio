from torch.utils.data import Sampler
import numpy as np

from data_module.dataset import TaskGroupedAudioGenConcatDataset


class TaskIteratingSampler(Sampler):
    def __init__(
        self,
        data_source: TaskGroupedAudioGenConcatDataset,
        shuffle: bool = True
    ):
        self.tasks = data_source.tasks
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
                data_source.task_to_cum_sum_lengths[task][-1]
            )
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
