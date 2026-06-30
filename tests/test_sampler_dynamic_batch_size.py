from torch.utils.data import Dataset

from data_module.batch_sampler import TaskIteratingDynamicBatchSampler
from data_module.dataset import TaskGroupedAudioGenConcatDataset
from data_module.sampler import TaskIteratingSampler

TASK_LENGTHS = [
    ("text_to_audio", [2, 4, 8]),
    ("text_to_music", [3, 5, 7]),
    ("video_to_audio", [2, 6, 8]),
    ("text_to_speech", [3, 4, 6]),
    ("speech_enhancement", [2, 5, 8]),
    ("audio_super_resolution", [3, 6, 7]),
    ("singing_voice_synthesis", [2, 4, 6]),
]


class DummyTaskDataset(Dataset):
    def __init__(self, task: str, lengths: list[int]):
        self.task = task
        self.lengths = lengths

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        return {"task": self.task, "length": self.lengths[idx]}

    def get_length(self, idx):
        return self.lengths[idx]


def test_task_iterating_dynamic_batch_sampler_print_lengths():
    dataset = TaskGroupedAudioGenConcatDataset(
        datasets=[
            DummyTaskDataset(task, lengths) for task, lengths in TASK_LENGTHS
        ]
    )
    sampler = TaskIteratingSampler(dataset, shuffle=False)
    batch_sampler = TaskIteratingDynamicBatchSampler(
        sampler=sampler,
        batch_size=4,
        batch_length_threshold=20,
        max_samples_per_batch=4,
    )

    batch_iter = iter(batch_sampler)
    for batch_idx in range(8):
        batch = next(batch_iter)
        sample_lengths = [
            dataset.get_length((task, idx)) for task, idx in batch
        ]
        padded_total_length = max(sample_lengths) * len(sample_lengths)

        print(
            f"batch={batch_idx}, samples={batch}, "
            f"sample_lengths={sample_lengths}, "
            f"padded_total_length={padded_total_length}"
        )

        assert padded_total_length <= batch_sampler.batch_length_threshold
        assert len(batch) <= batch_sampler.max_samples_per_batch


if __name__ == "__main__":
    test_task_iterating_dynamic_batch_sampler_print_lengths()
