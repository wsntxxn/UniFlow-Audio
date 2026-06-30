import torch
from tqdm import tqdm

from accelerate.data_loader import BatchSamplerShard

from data_module.dataset import (
    TextToAudioDataset, TextToSpeechDataset, VideoToAudioDataset,
    SpeechEnhancementDataset, TaskGroupedAudioGenConcatDataset
)
from data_module.collate_function import PaddingCollateWithAnyContent
from data_module.batch_sampler import TaskGroupedDynamicBatchSampler

dataset = TaskGroupedAudioGenConcatDataset(
    datasets=[
        TextToAudioDataset(
            content="./data/audiocaps_v2/train/caption.jsonl",
            audio="./data/audiocaps_v2/train/audio.jsonl",
            task_instruction="./data/instructions/t5_embeddings.h5",
            instruction_idx=1,
            target_sr=24000
        ),
        VideoToAudioDataset(
            content="./data/vggsound/mmaudio_clip_sync/train.jsonl",
            audio="./data/vggsound/clip/train/audio.jsonl",
            target_sr=24000,
            task_instruction="./data/instructions/t5_embeddings.h5",
            petrel_oss_config=
            "/mnt/shared-storage-user/xuxuenan/petreloss.conf",
            downsampling_ratio=480
        ),
        TextToSpeechDataset(
            content=[
                "./data/libritts_cast_tts/train_100.jsonl",
                "./data/libritts_cast_tts/train_360.jsonl",
                "./data/libritts_cast_tts/train_500.jsonl",
            ],
            audio=[
                "./data/libritts_cast_tts/audio.jsonl",
            ],
            content_col="phoneme",
            target_sr=24000,
            task_instruction="./data/instructions/t5_embeddings.h5",
        ),
        SpeechEnhancementDataset(
            content=
            "./data/speech_enhancement/libritts_360_wham/train/noisy_speech.jsonl",
            audio=
            "./data/speech_enhancement/libritts_360_wham/train/audio.jsonl",
            base_content_path=
            "s3://xuxuenan/data/uniflow_audio/speech_enhancement/Libritts_360+Wham",
            base_audio_path=
            "s3://xuxuenan/data/uniflow_audio/speech_enhancement/Libritts_360+Wham",
            downsampling_ratio=480,
            target_sr=24000,
            max_duration=5.0,
            task_instruction="./data/instructions/t5_embeddings.h5",
            petrel_oss_config="/mnt/shared-storage-user/xuxuenan/petreloss.conf"
        )
    ]
)
item = dataset.datasets[0][0]
batch_sampler = TaskGroupedDynamicBatchSampler(
    dataset,
    batch_length_threshold=512.0,
    max_samples_per_batch=64,
    random_seed=666,
)
collate_fn = PaddingCollateWithAnyContent(
    pad_keys=["waveform", "duration", "instruction"],
    content_pad_keys=[
        "phoneme", "phoneme_duration", "midi", "midi_duration", "is_slur",
        "frames", "prompt_waveform"
    ],
    time_aligned_tasks=[
        "text_to_speech", "singing_voice_synthesis", "speech_enhancement",
        "audio_super_resolution", "video_to_audio"
    ],
    non_time_aligned_tasks=[
        "text_to_speech", "singing_voice_synthesis", "text_to_audio",
        "text_to_music"
    ]
)
dataloader = torch.utils.data.DataLoader(
    dataset, collate_fn=collate_fn, num_workers=8, batch_sampler=batch_sampler
)

# for rank in range(4):
#     sharded = BatchSamplerShard(
#         sampler,
#         num_processes=4,
#         process_index=rank,
#         split_batches=False,
#         even_batches=False,
#     )
#     it = iter(sharded)
#     print('rank', rank, [next(it)[0][0] for _ in range(6)])

batch_idx = 0
for batch in tqdm(dataloader):
    print(batch["task"])
    batch_idx += 1

    # if batch_idx == 100:
    #     break
