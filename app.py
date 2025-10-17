#!/usr/bin/env python3

import gradio as gr
import os
from inference_cli import InferenceCLI

# Initialize inference CLI
cli = InferenceCLI()

# Available model choices
MODEL_CHOICES = [
    "UniFlow-Audio-large", "UniFlow-Audio-medium", "UniFlow-Audio-small"
]

# Default model name
DEFAULT_MODEL = "UniFlow-Audio-large"

# Pre-initialize models
print("Initializing models, please wait...")
print(f"Loading main model: {DEFAULT_MODEL}")
cli.init_model(DEFAULT_MODEL)

print("Loading speaker model for TTS...")
cli.init_speaker_model()

print("Loading G2P model for TTS...")
from montreal_forced_aligner.g2p.generator import PyniniConsoleGenerator
if not cli.g2p:
    cli.g2p = PyniniConsoleGenerator(
        g2p_model_path=cli.model.g2p_model_path,
        strict_graphemes=False,
        num_pronunciations=1,
        include_bracketed=False
    )
    cli.g2p.setup()

print("Loading SVS processor for singing voice synthesis...")
cli.init_svs_processor()

print("Loading video preprocessor for V2A...")
cli.init_video_preprocessor()

print("All models loaded successfully!")


def text_to_audio(
    caption,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Text to Audio generation"""
    output_path = "./outputs/t2a_output.wav"
    os.makedirs("./outputs", exist_ok=True)

    try:
        cli.t2a(
            caption=caption,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def text_to_music(
    caption,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Text to Music generation"""
    output_path = "./outputs/t2m_output.wav"
    os.makedirs("./outputs", exist_ok=True)

    try:
        cli.t2m(
            caption=caption,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def text_to_speech(
    transcript,
    ref_speaker_audio,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Text to Speech synthesis"""
    output_path = "./outputs/tts_output.wav"
    os.makedirs("./outputs", exist_ok=True)

    try:
        cli.tts(
            transcript=transcript,
            ref_speaker_speech=ref_speaker_audio,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def singing_voice_synthesis(
    singer,
    lyric,
    notes,
    note_durations,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Singing Voice Synthesis"""
    output_path = "./outputs/svs_output.wav"
    os.makedirs("./outputs", exist_ok=True)

    try:
        music_score = f"{lyric}<sep>{notes}<sep>{note_durations}"
        cli.svs(
            singer=singer,
            music_score=music_score,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def speech_enhancement(
    noisy_audio,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Speech Enhancement"""
    output_path = "./outputs/se_output.wav"
    os.makedirs("./outputs", exist_ok=True)

    try:
        cli.se(
            noisy_speech=noisy_audio,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Enhancement successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def audio_super_resolution(
    low_sr_audio,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Audio Super Resolution"""
    output_path = "./outputs/sr_output.wav"
    os.makedirs("./outputs", exist_ok=True)

    try:
        cli.sr(
            low_sr_audio=low_sr_audio,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Super-resolution successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def video_to_audio(
    video,
    model_name,
    guidance_scale,
    num_steps,
    progress=gr.Progress(track_tqdm=True)
):
    """Video to Audio generation"""
    output_path = "./outputs/v2a_output.mp4"
    os.makedirs("./outputs", exist_ok=True)

    try:
        cli.v2a(
            video=video,
            model_name=model_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            output_path=output_path
        )
        return output_path, "Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Create Gradio Interface
with gr.Blocks(
    title="UniFlow-Audio Inference Demo", theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# 🔊 UniFlow-Audio Inference Demo")
    gr.Markdown("Multi-task Audio Generation System based on UniFlow-Audio")

    with gr.Tabs():
        # Tab 1: Text to Audio
        with gr.Tab("📢 Text to Audio (T2A)"):
            with gr.Row():
                with gr.Column():
                    t2a_caption = gr.Textbox(
                        label="Audio Caption",
                        placeholder="e.g., a man is speaking then a dog barks",
                        lines=3
                    )
                    t2a_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        t2a_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5
                        )
                        t2a_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    t2a_button = gr.Button("Generate Audio", variant="primary")

                with gr.Column():
                    t2a_output = gr.Audio(
                        label="Generated Audio", type="filepath"
                    )
                    t2a_status = gr.Textbox(label="Status")

            t2a_button.click(
                fn=text_to_audio,
                inputs=[t2a_caption, t2a_model, t2a_guidance, t2a_steps],
                outputs=[t2a_output, t2a_status]
            )

            gr.Examples(
                examples=[
                    ["a man is speaking then a dog barks", 5.0, 25],
                    ["footsteps on wooden floor", 5.0, 25],
                ],
                inputs=[t2a_caption, t2a_guidance, t2a_steps]
            )

        # Tab 2: Text to Music
        with gr.Tab("🎼 Text to Music (T2M)"):
            with gr.Row():
                with gr.Column():
                    t2m_caption = gr.Textbox(
                        label="Music Caption",
                        placeholder="e.g., pop music with a male singing rap",
                        lines=3
                    )
                    t2m_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        t2m_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5
                        )
                        t2m_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    t2m_button = gr.Button("Generate Music", variant="primary")

                with gr.Column():
                    t2m_output = gr.Audio(
                        label="Generated Music", type="filepath"
                    )
                    t2m_status = gr.Textbox(label="Status")

            t2m_button.click(
                fn=text_to_music,
                inputs=[t2m_caption, t2m_model, t2m_guidance, t2m_steps],
                outputs=[t2m_output, t2m_status]
            )

            gr.Examples(
                examples=[
                    ["pop music with a male singing rap", 5.0, 25],
                    ["classical piano solo", 5.0, 25],
                ],
                inputs=[t2m_caption, t2m_guidance, t2m_steps]
            )

        # Tab 3: Text to Speech
        with gr.Tab("🗣️ Text to Speech (TTS)"):
            with gr.Row():
                with gr.Column():
                    tts_transcript = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="e.g., Hello this is a special sentence",
                        lines=3
                    )
                    tts_ref_audio = gr.Audio(
                        label="Reference Speaker Audio", type="filepath"
                    )
                    tts_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        tts_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5
                        )
                        tts_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    tts_button = gr.Button(
                        "Synthesize Speech", variant="primary"
                    )

                with gr.Column():
                    tts_output = gr.Audio(
                        label="Synthesized Speech", type="filepath"
                    )
                    tts_status = gr.Textbox(label="Status")

            tts_button.click(
                fn=text_to_speech,
                inputs=[
                    tts_transcript, tts_ref_audio, tts_model, tts_guidance,
                    tts_steps
                ],
                outputs=[tts_output, tts_status]
            )

            gr.Examples(
                examples=[
                    [
                        "Hello this is a special sentence with zyloph",
                        "./data/egs/tts_speaker_ref.wav", 5.0, 25
                    ],
                    [
                        "The quick brown fox jumps over the lazy dog",
                        "./data/egs/tts_speaker_ref.wav", 5.0, 25
                    ],
                ],
                inputs=[
                    tts_transcript, tts_ref_audio, tts_guidance, tts_steps
                ]
            )

        # Tab 4: Singing Voice Synthesis
        with gr.Tab("🎤 Singing Voice Synthesis (SVS)"):
            with gr.Row():
                with gr.Column():
                    svs_singer = gr.Dropdown(
                        label="Singer",
                        choices=[
                            "Alto-1", "Alto-2", "Alto-3", "Alto-4", "Alto-5",
                            "Alto-6", "Alto-7", "Bass-1", "Bass-2", "Bass-3",
                            "Soprano-1", "Soprano-2", "Soprano-3", "Tenor-1",
                            "Tenor-2", "Tenor-3", "Tenor-4", "Tenor-5",
                            "Tenor-6", "Tenor-7"
                        ],
                        value="Alto-2"
                    )
                    svs_lyric = gr.Textbox(
                        label="Lyrics",
                        placeholder="e.g., AP你要相信AP相信我们会像童话故事里AP",
                        lines=2
                    )
                    svs_notes = gr.Textbox(
                        label="Note Sequence",
                        placeholder="e.g., rest | G#3 | A#3 C4 | D#4 | ...",
                        lines=2
                    )
                    svs_durations = gr.Textbox(
                        label="Note Durations",
                        placeholder=
                        "e.g., 0.14 | 0.47 | 0.1905 0.1895 | 0.41 | ...",
                        lines=2
                    )
                    svs_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        svs_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5
                        )
                        svs_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    svs_button = gr.Button(
                        "Synthesize Singing", variant="primary"
                    )

                with gr.Column():
                    svs_output = gr.Audio(
                        label="Synthesized Singing", type="filepath"
                    )
                    svs_status = gr.Textbox(label="Status")

            svs_button.click(
                fn=singing_voice_synthesis,
                inputs=[
                    svs_singer, svs_lyric, svs_notes, svs_durations, svs_model,
                    svs_guidance, svs_steps
                ],
                outputs=[svs_output, svs_status]
            )

            gr.Examples(
                examples=[
                    [
                        "Alto-2", "AP你要相信AP相信我们会像童话故事里AP",
                        "rest | G#3 | A#3 C4 | D#4 | D#4 F4 | rest | E4 F4 | F4 | D#4 A#3 | A#3 | A#3 | C#4 | B3 C4 | C#4 | B3 C4 | A#3 | G#3 | rest",
                        "0.14 | 0.47 | 0.1905 0.1895 | 0.41 | 0.3005 0.3895 | 0.21 | 0.2391 0.1809 | 0.32 | 0.4105 0.2095 | 0.35 | 0.43 | 0.45 | 0.2309 0.2291 | 0.48 | 0.225 0.195 | 0.29 | 0.71 | 0.14",
                        5.0, 25
                    ],
                ],
                inputs=[
                    svs_singer, svs_lyric, svs_notes, svs_durations,
                    svs_guidance, svs_steps
                ]
            )

            gr.Markdown(
                """
            ### Usage Instructions
            - **Lyrics Format**: Use AP for pauses, e.g., `AP你要相信AP相信我们会像童话故事里AP`
            - **Note Format**: Separate with `|`, use spaces for simultaneous notes, use `rest` for rests
            - **Duration Format**: Note durations in seconds, separated by `|`
            """
            )

        # Tab 5: Speech Enhancement
        with gr.Tab("🔊 Speech Enhancement (SE)"):
            with gr.Row():
                with gr.Column():
                    se_input = gr.Audio(label="Noisy Speech", type="filepath")
                    se_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        se_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=1.0,
                            step=0.5
                        )
                        se_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    se_button = gr.Button("Enhance Speech", variant="primary")

                with gr.Column():
                    se_output = gr.Audio(
                        label="Enhanced Speech", type="filepath"
                    )
                    se_status = gr.Textbox(label="Status")

            se_button.click(
                fn=speech_enhancement,
                inputs=[se_input, se_model, se_guidance, se_steps],
                outputs=[se_output, se_status]
            )

            gr.Examples(
                examples=[
                    ["./data/egs/se_noisy_sample.wav", 1.0, 25],
                ],
                inputs=[se_input, se_guidance, se_steps]
            )

        # Tab 6: Audio Super Resolution
        with gr.Tab("⬆️ Audio Super Resolution (SR)"):
            with gr.Row():
                with gr.Column():
                    sr_input = gr.Audio(
                        label="Low Sample Rate Audio", type="filepath"
                    )
                    sr_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        sr_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=1.0,
                            step=0.5
                        )
                        sr_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    sr_button = gr.Button(
                        "Super-Resolve Audio", variant="primary"
                    )

                with gr.Column():
                    sr_output = gr.Audio(
                        label="High Sample Rate Audio", type="filepath"
                    )
                    sr_status = gr.Textbox(label="Status")

            sr_button.click(
                fn=audio_super_resolution,
                inputs=[sr_input, sr_model, sr_guidance, sr_steps],
                outputs=[sr_output, sr_status]
            )

            gr.Examples(
                examples=[
                    ["./data/egs/sr_low_sr_sample.wav", 1.0, 25],
                ],
                inputs=[sr_input, sr_guidance, sr_steps]
            )

        # Tab 7: Video to Audio
        with gr.Tab("🎬 Video to Audio (V2A)"):
            with gr.Row():
                with gr.Column():
                    v2a_input = gr.Video(label="Input Video")
                    v2a_model = gr.Dropdown(
                        label="Model Name",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL
                    )
                    with gr.Row():
                        v2a_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5
                        )
                        v2a_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=10,
                            maximum=100,
                            value=25,
                            step=1
                        )
                    v2a_button = gr.Button("Generate Audio", variant="primary")

                with gr.Column():
                    v2a_output = gr.Video(label="Video with Audio")
                    v2a_status = gr.Textbox(label="Status")

            v2a_button.click(
                fn=video_to_audio,
                inputs=[v2a_input, v2a_model, v2a_guidance, v2a_steps],
                outputs=[v2a_output, v2a_status]
            )

            gr.Examples(
                examples=[
                    ["./data/egs/v2a_video_sample.mp4", 5.0, 25],
                ],
                inputs=[v2a_input, v2a_guidance, v2a_steps]
            )

    gr.Markdown(
        """
    ---
    ### 📝 Notes
    - **Model Name**: Choose from `UniFlow-Audio-large`, `UniFlow-Audio-medium`, or `UniFlow-Audio-small`
    - **Guidance Scale**: Controls the guidance strength of the input condition on the output
    - **Sampling Steps**: Number of flow matching sampling steps
    
    💡 Tip: Models will be automatically downloaded on first run, please be patient
    """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
