import os
import sys
from importlib.resources import files

sys.path.append(os.getcwd() + "/src")

from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import StreamingHFDataset
from f5_tts.model.utils import get_tokenizer
from rl import trainer_rl

# -------------------------- Dataset Settings --------------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024

tokenizer = "custom"  # 'pinyin', 'char', or 'custom'
# if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
tokenizer_path = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
dataset_name = "LibriTTS_100"

# -------------------------- Training Settings -------------------------- #

exp_name = "F5TTS_v1_Base_rl"  # F5TTS_Base | E2TTS_Base

learning_rate = 1e-5

batch_size_per_gpu = 2  # 8 GPUs, 8 * 38400 = 307200, 1500 * 256 / 24000 = 16s
batch_size_type = "sample"  # "frame" or "sample"
# batch_size_per_gpu = 1500  # 8 GPUs, 8 * 38400 = 307200, 1500 * 256 / 24000 = 16s
# batch_size_type = "frame"  # "frame" or "sample"
min_samples = 2  # min sequences per batch if use frame-wise batch_size.
max_samples = 4  # max sequences per batch if use frame-wise batch_size.
grad_accumulation_steps = 2  # note: #updates = #steps / grad_accumulation_steps
max_grad_norm = 1.0

total_updates = 5000
num_warmup_updates = 100  # warmup steps
save_per_updates = 1  # save checkpoint per steps

# model params
wandb_resume_id = None
model_cls = DiT
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

pretrain_ckpt = "ckpts/F5TTS_v1_Base/model_last.pt"

# save checkpoint to ckpts/exp_name
checkpoint_path=str(files("f5_tts").joinpath(f"../../ckpts/{exp_name}"))

# ----------------------------------------------------------------------- #


def main():
    if tokenizer == "custom":
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    else:
        vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
    )

    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    if pretrain_ckpt != "":
        model = load_checkpoint(model, pretrain_ckpt, "cpu", use_ema=True)

    trainer = trainer_rl.GRPOTrainer(
        model,
        total_updates,
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        keep_last_n_checkpoints=1,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        min_samples=min_samples,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=True,
    )

    device = trainer.accelerator.device
    print(f"Using device: {device}")

    # from f5_tts.model.dataset import load_dataset
    # train_dataset = load_dataset(dataset_name, "char", mel_spec_kwargs=mel_spec_kwargs)

    # from datasets import load_dataset
    # train_dataset = load_dataset(
    #     "blabble-io/libritts_r", "clean", split="train.clean.100", streaming=True)
    # train_dataset = StreamingHFDataset(train_dataset, **mel_spec_kwargs)

    from datasets import load_dataset, interleave_datasets
    en = load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"en": "Emilia/EN/*.tar"},
        split="en",
        streaming=True
    )
    zh = load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"zh": "Emilia/ZH/*.tar"},
        split="zh",
        streaming=True
    )
    train_dataset = interleave_datasets([en, zh], probabilities=[0.5, 0.5], seed=42)
    train_dataset = StreamingHFDataset(train_dataset)

    trainer.train(
        train_dataset,
        num_workers=8,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
