import os
import json
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter

from utils.phonemize import sentence_to_phones, text_norm


def convert_sentence(sentence, word2phones, g2p_model):
    phones, oov_list = sentence_to_phones(sentence, word2phones, g2p_model)
    return " ".join(phones), oov_list


# ---------------- OOV statistics ----------------
def process_oov_single(args):
    lab_file, word2phones = args
    try:
        with open(lab_file, "r", encoding="utf-8") as f:
            sentence_raw = f.read().strip()

        sentence_norm = text_norm(sentence_raw)
        _, oov_list = sentence_to_phones(sentence_raw, word2phones)

        return {
            "file": lab_file,
            "sentence_raw": sentence_raw,
            "sentence_norm": sentence_norm,
            "oov": oov_list
        }
    except Exception:
        return {
            "file": lab_file,
            "sentence_raw": "",
            "sentence_norm": "",
            "oov": []
        }


def validate_oov_parallel(root_dir, word2phones, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    lab_files = [
        os.path.join(dp, f) for dp, _, files in os.walk(root_dir)
        for f in files if f.endswith(".lab")
    ]

    print(
        f"🔍 Scanning OOV, {len(lab_files)} files in total, {num_workers} processes in parallel"
    )

    oov_counter = Counter()
    oov_details = []
    args_list = [(f, word2phones) for f in lab_files]

    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_oov_single, args_list),
            total=len(args_list)
        ):
            if result["oov"]:
                oov_counter.update(result["oov"])
                oov_details.append(result)

    print(f"👉 Total OOV number: {sum(oov_counter.values())}")
    print(f"👉 Total OOV number in detail: {len(oov_counter)}")

    save_oov_path = "oov_words.json"
    with open(save_oov_path, "w", encoding="utf-8") as f:
        json.dump(oov_counter, f, ensure_ascii=False, indent=4)

    save_details_path = "oov_details.json"
    with open(save_details_path, "w", encoding="utf-8") as f:
        json.dump(oov_details, f, ensure_ascii=False, indent=4)

    return oov_counter, oov_details


# ---------------- Process lab files ----------------
def process_single_lab(args):
    lab_file, word2phones = args
    try:
        with open(lab_file, "r", encoding="utf-8") as f:
            sentence = f.read().strip()

        phone_str, _ = convert_sentence(sentence, word2phones)
        output_file = lab_file.replace(".lab", ".dict.phone")

        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(phone_str + "\n")
        return True
    except Exception as e:
        return f"❌ {lab_file}: {e}"


def process_lab_files_parallel(root_dir, word2phones, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    lab_files = [
        os.path.join(dp, f) for dp, _, files in os.walk(root_dir)
        for f in files if f.endswith(".lab")
    ]

    print(f"📊 Total {len(lab_files)} files, using {num_workers} processes")

    failures = []
    args_list = [(f, word2phones) for f in lab_files]

    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_lab, args_list),
            total=len(args_list)
        ):
            if result is not True:
                failures.append(result)

    print("✅ All files processed")
    if failures:
        print(f"⚠️ {len(failures)} files failed (examples):")
        for msg in failures[:5]:
            print(" ", msg)


# ========================= Main function =========================
if __name__ == "__main__":
    from montreal_forced_aligner.g2p.generator import PyniniConsoleGenerator
    # Initialize G2P
    g2p = PyniniConsoleGenerator(
        g2p_model_path='english_us_arpa',
        strict_graphemes=False,
        num_pronunciations=1,
        include_bracketed=False
    )
    g2p.setup()

    word2phones = json.load(
        open("data/word2phone.json", "r", encoding="utf-8")
    )

    # sentence = """You'll see in a moment what the difference is between 'convenient' and 'inconvenient.' You quite understand it now, don't you?" """
    sentence = """Stagecraft is the art of getting over these and other difficulties, and (if possible) getting over them in a showy manner, so that people will say, "How remarkable his stagecraft is for so young a writer," when otherwise they mightn't have noticed it at all."""
    print("sentence:", sentence)
    phone, OOV = convert_sentence(sentence, word2phones, g2p)
    print("norm sentence:", text_norm(sentence))
    print("→ Phone:", phone)
    print("→ OOV:", OOV)

    # root_dir = "/hpc_stor03/sjtu_home/jiahao.mei/data/LibriSpeech-pc/test-clean"

    # # validate_oov_parallel(root_dir, word2phones)

    # process_lab_files_parallel(root_dir, word2phones)

    # print(f'Number of Out-of-Vocabulary words that G2P also failed to handle: {g2p_error_cnt}, {g2p_error_list}')
