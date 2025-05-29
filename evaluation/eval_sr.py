import json
from pathlib import Path
import argparse
from tqdm import tqdm
from ssr_eval.metrics import AudioMetrics  


def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["audio_id"]] = item["audio"]
    return data


def evaluate(args):
    ref_dict = load_jsonl(args.ref_audio_jsonl)
    gen_dict = load_jsonl(args.gen_audio_jsonl)

    evaluator = AudioMetrics(rate=24000)  # or use your  args.sr，

    total_lsd = 0.0
    count = 0

    for audio_id in tqdm(gen_dict.keys()):
        if audio_id not in ref_dict:
            print(f"[WARN] audio_id {audio_id} not found in generated dict, skipping...")
            continue

        ref_path = ref_dict[audio_id]
        gen_path = gen_dict[audio_id]

        try:
            metrics = evaluator.evaluation(gen_path, ref_path, gen_path)
            total_lsd += metrics["lsd"]
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed on {audio_id}: {e}")
            continue

    if count == 0:
        print("not run eval")
        return

    avg_lsd = total_lsd / count
    print(f"\n eval  {count}  audio samples")
    print(f" mean LSD: {avg_lsd:.6f}")

    # 输出 JSON 文件
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"avg_lsd": avg_lsd, "num_samples": count}, f, indent=2)
    print(f"result save to : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_audio_jsonl",
        "-r",
        type=str,
        required=True,
        help="path to reference audio jsonl file"
    )
    parser.add_argument(
        "--gen_audio_jsonl",
        "-g",
        type=str,
        required=True,
        help="path to generated audio jsonl file"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="path to output file"
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing (not used yet)"
    )

    args = parser.parse_args()
    evaluate(args)
