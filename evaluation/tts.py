import os
import json
import argparse
import string
import numpy as np
from pathlib import Path
from typing import Sequence

import librosa
import torch
import torch.nn.functional as F
import torchaudio
import utmosv2
from accel_hydra.utils.general import read_jsonl_to_mapping
from jiwer import compute_measures
from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecSpeakerLabelModel
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer
from zhon.hanzi import punctuation

if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    from huggingface_hub import snapshot_download

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

english_normalizer = EnglishTextNormalizer()


def get_audio_duration(filepath):
    metadata = torchaudio.info(filepath)
    num_frames = metadata.num_frames
    sample_rate = metadata.sample_rate
    duration = num_frames / sample_rate
    return duration


def load_asr_model(model_name, lang='en', ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel
        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            disable_update=True,
        )
    elif lang == "en":
        if model_name == "whisper":  # requires numpy==2.2
            from faster_whisper import WhisperModel
            model_name_or_path = "large-v3" if ckpt_dir == "" else ckpt_dir
            model = WhisperModel(
                model_name_or_path, device="cuda", compute_type="float16"
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


def get_generated_audio_mapping(directory: str, exts: Sequence = [".wav"]):
    audio_mapping = {}
    for fpath in Path(directory).iterdir():
        if fpath.suffix.lower() in exts:
            audio_id = fpath.stem
            if "_" in audio_id:
                audio_id = audio_id.split("_")[0]
            audio_mapping[audio_id] = fpath.as_posix()
    return audio_mapping


# def get_libritts_text_mapping(txt_root):
#     """
#     Traverse all normalized.txt files under LibriTTS test-clean and build a mapping.
#     Example: '5683_32865_000001_000000' -> 'reference text'
#     """
#     mapping = {}
#     for root, _, files in os.walk(txt_root):
#         for f in files:
#             if f.endswith(".normalized.txt"):
#                 utt_id = f.replace(".normalized.txt", "")
#                 path = os.path.join(root, f)
#                 with open(path, "r") as t:
#                     content = t.read().strip()
#                     mapping[utt_id] = content
#     return mapping

# def get_reference_audio_mapping(txt_root):
#     """
#     Locate reference audio files in LibriTTS test-clean.
#     Assumes .wav reference files exist in the same directory as normalized.txt.
#     Returns: {'5683_32865_000001_000000': '/path/to/ref.wav'}
#     """
#     mapping = {}
#     for root, _, files in os.walk(txt_root):
#         for f in files:
#             if f.endswith(".wav"):
#                 utt_id = f.replace(".wav", "")
#                 ref_wav = os.path.join(root, f)
#                 if os.path.exists(ref_wav):
#                     mapping[utt_id] = ref_wav
#     return mapping


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


def load_utmos_model():
    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    model = model.to(DEVICE)
    return model


def load_utmos_v2_model():
    assert "UTMOSV2_CKPT" in os.environ, "Please set UTMOSV2_CKPT environment variable to the checkpoint path"
    model = utmosv2.create_model(
        pretrained=True,
        checkpoint_path=os.environ["UTMOSV2_CKPT"],
    )
    return model


def evaluate_tts(
    audio_dir_or_jsonl: str,
    ref_transcript_path: str,
    ref_audio_path: str,
    output_path: str,
    asr_model_name: str,
    speaker_model_name: str,
    lang: str = "en",
    asr_ckpt_path: str = "",
    num_workers: int = 12,
) -> None:
    assert lang in ("zh", "en")

    print("Loading ASR model...")
    asr_model = load_asr_model(asr_model_name, lang, asr_ckpt_path)

    print("Loading speaker embedding model...")
    speaker_model = load_speaker_model(speaker_model_name)

    print("Loading UTMOS model...")
    utmos_model = load_utmos_model()

    print("Loading UTMOS v2 model...")
    utmos_v2_model = load_utmos_v2_model()

    print("Building generated audio map...")
    if Path(audio_dir_or_jsonl).is_dir():
        aid_to_gen_audio = get_generated_audio_mapping(audio_dir_or_jsonl)
    elif audio_dir_or_jsonl.endswith(".jsonl"):
        aid_to_gen_audio = read_jsonl_to_mapping(
            audio_dir_or_jsonl, "audio_id", "audio"
        )
    print(f"Found {len(aid_to_gen_audio)} audio files")

    print("Building reference transcript map...")
    with open(ref_transcript_path, 'r') as f:
        aid_to_transcription = json.load(f)

    print("Building reference audio map...")
    if ref_audio_path.endswith(".json"):
        with open(ref_audio_path, 'r') as f:
            aid_to_ref_audio = json.load(f)
    elif ref_audio_path.endswith(".jsonl"):
        aid_to_ref_audio = read_jsonl_to_mapping(
            ref_audio_path, "audio_id", "audio"
        )

    total_word_errors = 0.0
    total_words = 0
    similarities = []
    utmos_scores = []
    results = []

    punctuation_all = punctuation + string.punctuation

    for audio_id in tqdm(aid_to_gen_audio, desc="Evaluating audio files"):
        gen_audio = aid_to_gen_audio[audio_id]
        gen_duration = get_audio_duration(gen_audio)
        if gen_duration < 0.5:
            print(f"Skipping {gen_audio}, duration less than 0.5s")
            continue

        transcription = aid_to_transcription[audio_id]
        if lang == "en":
            if asr_model_name == 'whisper':
                segments, _ = asr_model.transcribe(
                    gen_audio, beam_size=5, language="en"
                )
                pred_text = ""
                for segment in segments:
                    pred_text += " " + segment.text
            elif asr_model_name == 'nemo':
                pred_text = asr_model.transcribe([gen_audio],
                                                 verbose=False)[0].text.strip()

        raw_truth = transcription
        raw_hypo = pred_text

        for x in punctuation_all:
            transcription = transcription.replace(x, "")
            pred_text = pred_text.replace(x, "")

        transcription = transcription.replace("  ", " ")
        pred_text = pred_text.replace("  ", " ")

        if lang == "zh":
            transcription = " ".join([x for x in transcription])
            pred_text = " ".join([x for x in pred_text])
        elif lang == "en":
            transcription = english_normalizer(transcription)
            pred_text = english_normalizer(pred_text)

            transcription = transcription.lower()
            pred_text = pred_text.lower()

        ref_len = len(transcription.split())
        measures = compute_measures(transcription, pred_text)
        wer = measures["wer"]

        total_word_errors += wer * ref_len
        total_words += ref_len

        if speaker_model_name == 'titanet':
            pred_emb = speaker_model.get_embedding(gen_audio)
            ref_emb = speaker_model.get_embedding(aid_to_ref_audio[audio_id])
        elif speaker_model_name == 'ecapa_tdnn':
            with torch.no_grad():
                wav1, sr1 = torchaudio.load(gen_audio)
                wav2, sr2 = torchaudio.load(aid_to_ref_audio[audio_id])
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

        wav, sr = librosa.load(gen_audio, mono=True, sr=None)
        wav = torch.as_tensor(wav).to(DEVICE).unsqueeze(0)
        utmos_score = utmos_model(wav, sr).item()
        utmos_scores.append(utmos_score)

        line = {
            "audio_id": audio_id,
            "audio": gen_audio,
            "ref_audio": aid_to_ref_audio[audio_id],
            "reference": raw_truth,
            "prediction": raw_hypo,
            "WER": round(wer, 3),
            "SIM": round(sim, 3),
            "UTMOS": round(utmos_score, 3),
        }
        results.append(line)

    avg_wer = total_word_errors / total_words if total_words > 0 else 0.0
    avg_sim = np.mean(similarities) if similarities else 0.0
    avg_utmos = np.mean(utmos_scores) if utmos_scores else 0.0

    assert Path(audio_dir_or_jsonl).is_dir()
    utmosv2_res = utmos_v2_model.predict(
        input_dir=audio_dir_or_jsonl, batch_size=1, num_workers=num_workers
    )
    avg_utmos_v2 = np.mean([
        score_item["predicted_mos"] for score_item in utmosv2_res
    ])

    output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        for r in results:
            json.dump(r, f)
            f.write('\n')
        json.dump({
            "average_wer": avg_wer,
            "average_cosine_similarity": avg_sim,
            "average_utmos": avg_utmos,
            "average_utmos_v2": avg_utmos_v2,
        }, f)
        f.write('\n')

    print(f"Evaluation done: {len(results)} samples")
    print(f"Average WER (weighted): {avg_wer}")
    print(f"Average speaker similarity: {avg_sim}")
    print(f"Average UTMOS: {avg_utmos}")
    print(f"Average UTMOS v2: {avg_utmos_v2}")
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
        '--asr_model_name',
        type=str,
        default='nemo',
        choices=['whisper', 'nemo'],
        help='Name of English ASR model to use'
    )
    parser.add_argument(
        '--asr_ckpt_path',
        type=str,
        default='',
        help='Path to ASR checkpoint',
    )
    parser.add_argument(
        '--speaker_model_name',
        type=str,
        default='titanet',
        choices=['titanet', 'ecapa_tdnn'],
        help='Name of speaker embedding model to use'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='en',
        choices=['zh', 'en'],
        help='Language of the TTS model'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
        help='Number of workers for UTMOS v2 evaluation'
    )

    args = parser.parse_args()

    evaluate_tts(
        args.audio_dir_or_jsonl,
        args.ref_transcript_path,
        args.ref_audio_path,
        args.output_path,
        args.asr_model_name,
        args.speaker_model_name,
        args.lang,
        args.asr_ckpt_path,
        args.num_workers,
    )
