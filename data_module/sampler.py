import numpy as np
from torch.utils.data import Sampler

from data_module.dataset import TaskGroupedAudioGenConcatDataset


class TaskIteratingSampler(Sampler):
    def __init__(
        self,
        data_source: TaskGroupedAudioGenConcatDataset,
        shuffle: bool = True,
        task_sampling_weights: dict = None
    ):
        self.data_source = data_source
        self.tasks = data_source.tasks
        # add extra task sampling times for some tasks
        if task_sampling_weights:
            for key, weight in task_sampling_weights.items():
                if key in self.tasks:
                    self.tasks += [key] * (weight - 1)
        print(f'task sample order: {self.tasks}')
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


class InferenceTaskIteratingSampler(Sampler):
    # Finite sampler for inference
    def __init__(
        self,
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
