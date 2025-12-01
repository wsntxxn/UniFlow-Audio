import argparse
import json
from pathlib import Path


def main(args):
    librispeech_dir = Path(args.librispeech_dir)
    target_librispeech_dir = Path(args.target_librispeech_dir)

    target_librispeech_dir.mkdir(parents=True, exist_ok=True)

    audio_ids = set()
    gen_audio_ids = []
    with open(args.cross_sentence_meta, "r") as reader, \
        open(target_librispeech_dir / "ref_transcription.json", "w") as writer, \
        open(target_librispeech_dir / "prompt_audio.jsonl", "w") as prompt_writer:
        ref_transcription = {}
        for line in reader.readlines():
            prompt_audio_id, prompt_duration, prompt_text, \
                audio_id, duration, text = line.strip().split("\t")
            audio_ids.add(prompt_audio_id)
            audio_ids.add(audio_id)
            gen_audio_ids.append(audio_id)
            ref_transcription[audio_id] = text

            part1, part2, part3 = prompt_audio_id.split("-")
            prompt_audio_path: Path = librispeech_dir / "test-clean" / part1 / part2 / f"{prompt_audio_id}.flac"
            prompt_writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "audio": prompt_audio_path.as_posix()
                }) + "\n"
            )

        json.dump(ref_transcription, writer, indent=2)

    with open(target_librispeech_dir / "audio.jsonl", "w") as writer:
        for audio_id in audio_ids:
            part1, part2, part3 = audio_id.split("-")
            audio_path: Path = librispeech_dir / "test-clean" / part1 / part2 / f"{audio_id}.flac"
            writer.write(
                json.dumps({
                    "audio_id": audio_id,
                    "audio": audio_path.as_posix()
                }) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_dir",
        "-l",
        type=str,
        default="/mnt/shared-storage-user/brainllm-share/data/LibriSpeech"
    )
    parser.add_argument(
        "--cross_sentence_meta",
        type=str,
        default=
        "/mnt/shared-storage-user/xuxuenan/workspace/f5tts/data/librispeech_pc_test_clean_cross_sentence.lst"
    )
    parser.add_argument(
        "--target_librispeech_dir",
        "-t",
        type=str,
        default="./data/librispeech_pc"
    )
    args = parser.parse_args()

    main(args)
