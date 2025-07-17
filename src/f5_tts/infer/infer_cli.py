import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from cached_path import cached_path
from vocos import Vocos

sys.path.append(os.getcwd() + "/src")

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="F5TTS_v1_Base",
    help="F5TTS_v1_Base | F5TTS_Base | E2TTS_Base",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    default="",
    help="The Checkpoint .pt",
)
parser.add_argument("-r", "--ref_audio", type=str, required=True, help="Reference audio file < 15 seconds.")
parser.add_argument("-s", "--ref_text", type=str, required=True, help="Subtitle for the reference audio.")
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    required=True,
    help="Text to generate.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="tests",
    help="Path to output folder..",
)
parser.add_argument(
    "--remove_silence",
    default=False,
    help="Remove silence.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Adjust the speed of the audio generation (default: 1.0)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    help="Specify the device to run on",
)
args = parser.parse_args()

ref_audio = args.ref_audio
ref_text = args.ref_text
gen_text = args.gen_text

output_dir = args.output_dir
model = args.model
ckpt_file = args.ckpt_file
remove_silence = args.remove_silence
speed = args.speed
device = args.device
wave_path = Path(output_dir) / "infer_cli_out.wav"
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"

vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

# model params
if model == "F5TTS_v1_Base":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
elif model == "F5TTS_Base":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, text_mask_padding=False,
                     pe_attn_head=1)
elif model == "F5TTS_Small":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
elif model == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
elif model == "E2TTS_Small":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=768, depth=20, heads=12, ff_mult=4)

# load models
repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
# override for previous models
if model == "F5TTS_Base":
    ckpt_step = 1200000
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))

print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, device=device)


def main_process(ref_audio, ref_text, gen_text, model_obj, remove_silence, speed):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)

    audio, final_sample_rate, spectrogram = infer_process(
        ref_audio, ref_text, gen_text, model_obj, vocoder, speed=speed
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(wave_path, "wb") as f:
        sf.write(f.name, audio, final_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(f.name)
        print(f.name)


def main():
    main_process(ref_audio, ref_text, gen_text, ema_model, remove_silence, speed)


if __name__ == "__main__":
    main()
