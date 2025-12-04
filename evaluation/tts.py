# python evaluation/tts.py  \
#   --audio_dir '/cpfs_shared/jiahao.mei/code/x_to_audio_generation/experiments/voiceflow_infer' \
#   --exp_name voiceflow

# python evaluation/tts.py  \
#     --audio_dir '/cpfs_shared/jiahao.mei/code/x_to_audio_generation/experiments/voiceflow_infer' \
#     --libritts_txt_dir '/cpfs_shared/jiahao.mei/data/tts/LibriTTS' \
#     --exp_name voiceflow

from pathlib import Path
import os
import json
import argparse
import string
import numpy as np

import jiwer
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer
from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecSpeakerLabelModel
if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    from huggingface_hub import snapshot_download

from utils.general import read_jsonl_to_mapping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

english_normalizer = EnglishTextNormalizer()

# asr model: https://huggingface.co/nvidia/stt_en_conformer_transducer_xlarge


# speaker model: https://huggingface.co/nvidia/speakerverification_en_titanet_large
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
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            save_restore_connector = SaveRestoreConnector()
            nemo_dir = snapshot_download(
                "nvidia/stt_en_conformer_transducer_xlarge"
            )
            model = EncDecRNNTBPEModel.restore_from(
                restore_path=os.path.join(
                    nemo_dir, "stt_en_conformer_transducer_xlarge.nemo"
                ),
                override_config_path=None,
                map_location=None,
                strict=True,
                return_config=False,
                trainer=None,
                save_restore_connector=save_restore_connector
            )
        else:
            model = EncDecRNNTBPEModel.from_pretrained(
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


def load_speaker_model(model_name: str):
    if model_name == "titanet":
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            save_restore_connector = SaveRestoreConnector()
            nemo_dir = snapshot_download(
                "nvidia/speakerverification_en_titanet_large"
            )
            speaker_model = EncDecSpeakerLabelModel.restore_from(
                restore_path=os.path.join(
                    nemo_dir, "speakerverification_en_titanet_large.nemo"
                ),
                override_config_path=None,
                map_location=None,
                strict=True,
                return_config=False,
                trainer=None,
                save_restore_connector=save_restore_connector
            )
        else:
            speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                "nvidia/speakerverification_en_titanet_large"
            )
    elif model_name == "ecapa_tdnn":
        from ecapa_tdnn import ECAPA_TDNN_SMALL

        wavlm_ckpt_path = os.environ.get(
            "WAVLM_ECAPA_TDNN_CKPT",
            os.path.join(
                os.environ.get(
                    "TORCH_HOME", os.path.expanduser("~/.cache/torch")
                ), "hub/UniSpeech/wavlm_large_finetune.pth"
            )
        )
        speaker_model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=None
        )

        state_dict = torch.load(
            wavlm_ckpt_path,
            weights_only=True,
            map_location=lambda storage, loc: storage
        )
        speaker_model.load_state_dict(state_dict["model"], strict=False)
        speaker_model.eval()
        speaker_model.to(DEVICE)
    return speaker_model


def evaluate_tts(
    audio_dir_or_jsonl: str,
    libritts_txt_dir: str,
    ref_transcript_path: str,
    ref_audio_path: str,
    output_path: str,
    model_name: str,
    exp_name: str,
    speaker_model_name: str,
) -> None:
    print("Loading ASR model...")
    asr_model = load_asr_model(model_name)

    print("Loading speaker embedding model...")
    speaker_model = load_speaker_model(speaker_model_name)

    print("Building reference transcript map...")
    if not Path(ref_transcript_path).exists():
        ref_map = get_libritts_text_mapping(libritts_txt_dir)
        with open(ref_transcript_path, 'w') as f:
            json.dump(ref_map, f, indent=2)
    else:
        with open(ref_transcript_path, 'r') as f:
            ref_map = json.load(f)

    print("Building reference audio map...")
    if not Path(ref_audio_path).exists():
        ref_audio_map = get_reference_audio_mapping(libritts_txt_dir)
        with open(ref_audio_path, 'w') as f:
            json.dump(ref_audio_map, f, indent=2)
    else:
        if ref_audio_path.endswith(".json"):
            with open(ref_audio_path, 'r') as f:
                ref_audio_map = json.load(f)
        elif ref_audio_path.endswith(".jsonl"):
            ref_audio_map = read_jsonl_to_mapping(
                ref_audio_path, "audio_id", "audio"
            )

    if Path(audio_dir_or_jsonl).is_dir():
        audio_files = get_all_audio_files(audio_dir_or_jsonl)
    elif audio_dir_or_jsonl.endswith(".jsonl"):
        aid_to_audio = read_jsonl_to_mapping(
            audio_dir_or_jsonl, "audio_id", "audio"
        )
        audio_files = aid_to_audio.values()

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
        # try:
        if model_name == 'whisper':
            segments, _ = asr_model.transcribe(
                wav_path, beam_size=5, language="en"
            )
            pred_text = ""
            for segment in segments:
                pred_text += " " + segment.text
        elif model_name == 'nemo':
            pred_text = asr_model.transcribe([wav_path],
                                             verbose=False)[0].text.strip()

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

        if speaker_model_name == 'titanet':
            pred_emb = speaker_model.get_embedding(wav_path)
            ref_emb = speaker_model.get_embedding(ref_audio_map[utt_id])
        elif speaker_model_name == 'ecapa_tdnn':
            with torch.no_grad():
                wav1, sr1 = torchaudio.load(wav_path)
                wav2, sr2 = torchaudio.load(ref_audio_map[utt_id])
                if sr1 != 16000:
                    wav1 = torchaudio.functional.resample(
                        wav1, orig_freq=sr1, new_freq=16000
                    )
                wav1 = wav1.to(DEVICE)
                if sr2 != 16000:
                    wav2 = torchaudio.functional.resample(
                        wav2, orig_freq=sr2, new_freq=16000
                    )
                wav2 = wav2.to(DEVICE)

                pred_emb = speaker_model(wav1)
                ref_emb = speaker_model(wav2)

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

        # except Exception as e:
        #     print(f"Skipping file {wav_path}, reason: {e}")

    avg_wer = total_word_errors / total_words if total_words > 0 else 0.0
    avg_sim = np.mean(similarities) if similarities else 0.0

    if output_path == '':
        output_path = Path(
            './evaluation/result'
        ) / f'tts_results_{model_name}_{exp_name}.jsonl'
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        for r in results:
            json.dump(r, f)
            f.write('\n')
        json.dump({
            "average_wer": avg_wer,
            "average_cosine_similarity": avg_sim
        }, f)
        f.write('\n')

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
        '--audio_dir_or_jsonl',
        type=str,
        required=True,
        help=
        'Directory or JSONL filecontaining TTS-generated audio files (recursive search)'
    )
    parser.add_argument(
        '--libritts_txt_dir',
        type=str,
        default='/cpfs_shared/jiahao.mei/data/tts/LibriTTS/test-clean',
        help='Directory for LibriTTS test-clean files'
    )
    parser.add_argument(
        '--ref_transcript_path',
        type=str,
        default='./data/libritts/voiceflow_test/ref_transcript.json',
        help='Path to reference transcript JSON file'
    )
    parser.add_argument(
        '--ref_audio_path',
        type=str,
        default='./data/libritts/voiceflow_test/ref_audio.json',
        help='Path to reference audio JSON file'
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
        '--speaker_model_name',
        type=str,
        default='titanet',
        choices=['titanet', 'ecapa_tdnn'],
        help='Name of speaker embedding model to use'
    )
    parser.add_argument(
        '--exp_name', type=str, default='', help='Experiment name'
    )

    args = parser.parse_args()

    evaluate_tts(
        args.audio_dir_or_jsonl,
        args.libritts_txt_dir,
        args.ref_transcript_path,
        args.ref_audio_path,
        args.output_path,
        args.model_name,
        args.exp_name,
        args.speaker_model_name,
    )
