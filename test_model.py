from models.content_encoder.midi_encoder import FastSpeech2PitchEncoder
from data_module.dataset import PopCsSingingDataset
from data_module.collate_function import PaddingCollate

import torch

model = FastSpeech2PitchEncoder(
    phone_vocab_size=62,
    pitch_downsample_ratio=4,
    d_model=256,
    num_layers=4,
    num_heads=2,
    ffn_kernel_size=9,
    d_out=256
)
dataset = PopCsSingingDataset(
    content="./data/popcs/train/phone_pitch.jsonl",
    audio="./data/popcs/train/audio.jsonl",
    target_sr=24000,
    f0_stats="./data/popcs/train/f0_mean_std.npy",
    pitch_norm="log",
    use_uv=True
)

collate_fn = PaddingCollate(pad_keys=["waveform", "duration", "f0", "uv"])
dataloader = torch.utils.data.DataLoader(
    dataset, collate_fn=collate_fn, batch_size=4
)

batch = next(iter(dataloader))

out = model.encode_pitch(
    f0=batch["content"][0]["f0"].unsqueeze(0),
    uv=batch["content"][0]["uv"].unsqueeze(0)
)
print(out.shape)
print(batch["waveform"][0].shape[0] // 480)
