import json

# import fire
import h5py

with open("./data/vggsound/mmaudio_clip_sync/train.jsonl", "w") as writer:
    for idx in range(1, 6):
        h5_file = f"/mnt/shared-storage-gpfs2/speechllm-share/xuxuenan/workspace/uniflow_audio/data/vggsound/train/feat_{idx}.h5"
        with h5py.File(h5_file, "r") as hf:
            audio_ids = hf["audio_id"][()]
            for audio_id in audio_ids:
                writer.write(
                    json.dumps({
                        "audio_id": audio_id.decode(),
                        "video": h5_file,
                    }) + "\n"
                )

with open("./data/vggsound/mmaudio_clip_sync/val.jsonl", "w") as writer:
    h5_file = f"/mnt/shared-storage-gpfs2/speechllm-share/xuxuenan/workspace/uniflow_audio/data/vggsound/val/feat.h5"
    with h5py.File(h5_file, "r") as hf:
        audio_ids = hf["audio_id"][()]
        for audio_id in audio_ids:
            writer.write(
                json.dumps({
                    "audio_id": audio_id.decode(),
                    "video": h5_file,
                }) + "\n"
            )

with open("./data/vggsound/mmaudio_clip_sync/test.jsonl", "w") as writer:
    h5_file = f"/mnt/shared-storage-gpfs2/speechllm-share/xuxuenan/workspace/uniflow_audio/data/vggsound/test/feat.h5"
    with h5py.File(h5_file, "r") as hf:
        audio_ids = hf["audio_id"][()]
        for audio_id in audio_ids:
            writer.write(
                json.dumps({
                    "audio_id": audio_id.decode(),
                    "video": h5_file,
                }) + "\n"
            )
