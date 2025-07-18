import os
import sys
from cached_path import cached_path
from importlib.resources import files

sys.path.append(os.getcwd() + "/src")

from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import StreamingHFDataset
from f5_tts.model.utils import get_tokenizer

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

exp_name = "F5TTS_v1_Base"  # F5TTS_v1_Base | F5TTS_Base | F5TTS_Small | E2TTS_Base | E2TTS_Small

learning_rate = 7.5e-5

batch_size_per_gpu = 1500  # 8 GPUs, 8 * 38400 = 307200, 1500 * 256 / 24000 = 16s
batch_size_type = "frame"  # "frame" or "sample"
max_samples = 8  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
grad_accumulation_steps = 2  # note: #updates = #steps / grad_accumulation_steps
max_grad_norm = 1.0

total_updates = 1000
num_warmup_updates = 100  # warmup updates
save_per_updates = 100  # save checkpoint per updates

# model params
if exp_name == "F5TTS_v1_Base":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
    )
elif exp_name == "F5TTS_Base":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        text_mask_padding=False,
        pe_attn_head=1,
    )
elif exp_name == "F5TTS_Small":
    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(
        dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4
    )
elif exp_name == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(
        dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1
    )
elif exp_name == "E2TTS_Small":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=768, depth=20, heads=12, ff_mult=4)

# load models
repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
# override for previous models
if exp_name == "F5TTS_Base":
    ckpt_step = 1200000
elif exp_name == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

# ckpt_file = ""  # train from scratch
ckpt_file = str(
    cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.{ckpt_type}")
)

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

    if ckpt_file != "":
        ft_params = [
            "transformer.norm_out.linear.weight",
            "transformer.norm_out.linear.bias",
            "transformer.proj_out.weight",
            "transformer.proj_out.bias",
            "transformer.proj_out_ln_sig.weight",
            "transformer.proj_out_ln_sig.bias"
        ]

        for name, param in model.named_parameters():
            if name in ft_params:
                param.requires_grad = True
                print(f"Trainable: {name}")
            else:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    if ckpt_file != "":
        print("Setting learning rate to 1e-5")
        learning_rate = 1e-5
        model = load_checkpoint(model, ckpt_file, "cpu", use_ema=True)

    trainer = Trainer(
        model,
        total_updates,
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        keep_last_n_checkpoints=1,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=batch_size_per_gpu,
        batch_size_type=batch_size_type,
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

    from f5_tts.model.dataset import load_dataset
    train_dataset = load_dataset(dataset_name, "char", mel_spec_kwargs=mel_spec_kwargs)

    # from datasets import load_dataset
    # train_dataset = load_dataset(
    #     "blabble-io/libritts_r", "clean", split="train.clean.100", streaming=True)
    # train_dataset = StreamingHFDataset(train_dataset, **mel_spec_kwargs)

    # from datasets import load_dataset, interleave_datasets
    # en = load_dataset(
    #     "amphion/Emilia-Dataset",
    #     data_files={"en": "Emilia/EN/*.tar"},
    #     split="en",
    #     streaming=True
    # )
    # zh = load_dataset(
    #     "amphion/Emilia-Dataset",
    #     data_files={"zh": "Emilia/ZH/*.tar"},
    #     split="zh",
    #     streaming=True
    # )
    # train_dataset = interleave_datasets([en, zh], probabilities=[0.5, 0.5], seed=42)
    # train_dataset = StreamingHFDataset(train_dataset)

    trainer.train(
        train_dataset,
        num_workers=8,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
