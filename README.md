# F5R-TTS: Improving Flow-Matching based Text-to-Speech with Group Relative Policy Optimization

This is a simplified implementation of [F5R-TTS](https://github.com/FrontierLabs/F5R-TTS) based on the paper [F5R-TTS: Improving Flow-Matching based Text-to-Speech with Group Relative Policy Optimization](https://arxiv.org/abs/2504.02407), intended for learning purposes.

<div align="center">
<img width="200px" src="resource/overall.png" /><br>
<figcaption>Fig 1: The architecture of backbone.</figcaption>
</div>

<br>

<div align="center">
<img width="500px" src="resource/grpo_train.png" /><br>
<figcaption>Fig 2: The pipeline of GRPO phase.</figcaption>
</div>

<br>


## Installation

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n f5r-tts python=3.10
conda activate f5r-tts
pip install -r requirements.txt
```

## Inference

```bash
python ./src/f5_tts/infer/infer_cli.py \
  --model F5-TTS \
  --ckpt_file "your_model_path" \
  --ref_audio "path_to_reference.wav" \
  --ref_text "reference_text" \
  --gen_text "generated_text" \
  --output_dir ./tests
```

## Training

You need to download [SenseVoice_small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) and [wespeaker](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) for GRPO phase.

```bash
accelerate config

# Data preparing
python src/f5_tts/train/datasets/prepare_libritts.py

# Pretraining phase
accelerate launch src/f5_tts/train/train.py

# GRPO phase
accelerate launch src/f5_tts/train/train_rl.py
```

## [Evaluation](src/f5_tts/eval)
Follow the [README.md](src/f5_tts/eval/README.md) file.

## License

Our code is released under MIT License.

