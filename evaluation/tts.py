# python evaluation/tts.py  \
#   --audio_dir '/cpfs_shared/jiahao.mei/code/x_to_audio_generation/experiments/voiceflow_infer' \
#   --xp_name voiceflow

# python evaluation/tts.py  \
#     --audio_dir '/cpfs_shared/jiahao.mei/code/x_to_audio_generation/experiments/voiceflow_infer' \
#     --libritts_txt_dir '/cpfs_shared/jiahao.mei/data/tts/LibriTTS' \
#     --xp_name voiceflow

import os
import json
import jiwer
import argparse
import string
import numpy as np

from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer
import torch.nn.functional as F
import torchaudio
import nemo.collections.asr as nemo_asr

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

english_normalizer = EnglishTextNormalizer()

# asr model: https://huggingface.co/nvidia/stt_en_conformer_transducer_xlarge


# spkear model: https://huggingface.co/nvidia/speakerverification_en_titanet_large
def get_audio_duration(filepath):
    try:
        metadata = torchaudio.info(filepath)
        num_frames = metadata.num_frames
        sample_rate = metadata.sample_rate
        duration = num_frames / sample_rate
        return duration
    except Exception as e:
        print(f"Skipping file {filepath}, reason: {e}")
        return 0.0


def load_asr_model(model_name, lang='en', ckpt_dir=""):
    if model_name == "whisper":  # requires numpy==2.2
        if lang == "zh":
            from funasr import AutoModel
            model = AutoModel(
                model=os.path.join(ckpt_dir, "paraformer-zh"),
                disable_update=True,
            )  # following seed-tts setting
        elif lang == "en":
            from faster_whisper import WhisperModel
            model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
            model = WhisperModel(
                model_size, device="cuda", compute_type="float16"
            )
    elif model_name == "nemo":  # requires numpy<2.0
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            "nvidia/stt_en_conformer_transducer_xlarge"
        )

    return model


def get_all_audio_files(directory, exts={".wav"}):
    audio_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[-1].lower() in exts:
                audio_files.append(os.path.join(root, f))
    return audio_files


def get_libritts_text_mapping(txt_root):
    """
    Traverse all normalized.txt files under LibriTTS test-clean and build a mapping.
    Example: '5683_32865_000001_000000' -> 'reference text'
    """
    mapping = {}
    for root, _, files in os.walk(txt_root):
        for f in files:
            if f.endswith(".normalized.txt"):
                utt_id = f.replace(".normalized.txt", "")
                path = os.path.join(root, f)
                with open(path, "r") as t:
                    content = t.read().strip()
                    mapping[utt_id] = content
    return mapping


def get_reference_audio_mapping(txt_root):
    """
    Locate reference audio files in LibriTTS test-clean.
    Assumes .wav reference files exist in the same directory as normalized.txt.
    Returns: {'5683_32865_000001_000000': '/path/to/ref.wav'}
    """
    mapping = {}
    for root, _, files in os.walk(txt_root):
        for f in files:
            if f.endswith(".wav"):
                utt_id = f.replace(".wav", "")
                ref_wav = os.path.join(root, f)
                if os.path.exists(ref_wav):
                    mapping[utt_id] = ref_wav
    return mapping


def extract_utt_id(filepath):
    """Extract utterance ID like '5683_32865_000001_000000' from file path"""
    filename = os.path.basename(filepath)
    utt_id = os.path.splitext(filename)[0]
    while utt_id[0] == '0':
        utt_id = utt_id[1:]
    return utt_id


def evaluate_tts(
    audio_dir, libritts_txt_dir, output_path, model_name, xp_name
):
    print("Loading ASR model...")
    asr_model = load_asr_model(model_name)

    print("Loading speaker embedding model...")
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )

    print("Building reference transcript map...")
    ref_map = get_libritts_text_mapping(libritts_txt_dir)

    print("Building reference audio map...")
    ref_audio_map = get_reference_audio_mapping(libritts_txt_dir)

    audio_files = get_all_audio_files(audio_dir)
    print(f"Found {len(audio_files)} audio files")

    total_word_errors = 0.0
    total_words = 0
    similarities = []
    results = []

    for wav_path in tqdm(audio_files, desc="Evaluating audio files"):
        utt_id = extract_utt_id(wav_path)
        if utt_id not in ref_map:
            print(f"Skipping {wav_path}, reference text not found")
            continue
        if utt_id not in ref_audio_map:
            print(f"Skipping {wav_path}, reference audio not found")
            continue
        wav_dur = get_audio_duration(wav_path)
        if wav_dur < 0.5:
            print(f"Skipping {wav_path}, duration less than 0.5s")
            continue

        reference = ref_map[utt_id]
        try:
            if model_name == 'whisper':
                segments, _ = asr_model.transcribe(
                    wav_path, beam_size=5, language="en"
                )
                pred_text = ""
                for segment in segments:
                    pred_text += " " + segment.text
            elif model_name == 'nemo':
                pred_text = asr_model.transcribe([wav_path])[0].text.strip()

            for x in string.punctuation:
                pred_text = pred_text.replace(x, "")
                reference = reference.replace(x, "")

            pred_text = pred_text.replace("  ", " ")
            reference = reference.replace("  ", " ")
            pred_text = english_normalizer(pred_text)
            reference = english_normalizer(reference)

            reference_words = reference.lower().split()
            ref_len = len(reference_words)
            wer = jiwer.wer(reference.lower(), pred_text.lower())

            total_word_errors += wer * ref_len
            total_words += ref_len

            pred_emb = speaker_model.get_embedding(wav_path)
            ref_emb = speaker_model.get_embedding(ref_audio_map[utt_id])
            sim = F.cosine_similarity(pred_emb, ref_emb)[0].item()
            similarities.append(sim)

            line = {
                "utt_id": utt_id,
                "audio": wav_path,
                "ref_audio": ref_map[utt_id],
                "reference": reference,
                "prediction": pred_text,
                "WER": wer,
                "SIM": float(sim),
            }
            results.append(line)

        except Exception as e:
            print(f"Skipping file {wav_path}, reason: {e}")

    avg_wer = total_word_errors / total_words if total_words > 0 else 0.0
    avg_sim = np.mean(similarities) if similarities else 0.0

    if output_path == '':
        output_path = './evaluation/result/' + '_'.join([
            "tts_results", model_name, xp_name
        ]) + '.jsonl'

    with open(output_path, 'w') as f:
        for r in results:
            json.dump(r, f)
            f.write('\n')
        json.dump({"average_wer": avg_wer}, f)
        json.dump({"average_cosine_similarity": avg_sim}, f)

    print(
        f"\n✅ Evaluation done: {len(results)} samples, average WER (weighted): {avg_wer}"
    )
    print(
        f"\n✅ Evaluation done: {len(similarities)} samples, average Cosine similarity: {avg_sim}"
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help='Directory containing TTS-generated audio files (recursive search)'
    )
    parser.add_argument(
        '--libritts_txt_dir',
        type=str,
        default='/cpfs_shared/jiahao.mei/data/tts/LibriTTS/test-clean',
        help='Directory for LibriTTS test-clean files'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='',
        help='Output path for evaluation results'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='nemo',
        help='Name of ASR model to use'
    )
    parser.add_argument(
        '--xp_name', type=str, default='', help='Experiment name'
    )

    args = parser.parse_args()

    evaluate_tts(
        args.audio_dir, args.libritts_txt_dir, args.output_path,
        args.model_name, args.xp_name
    )
