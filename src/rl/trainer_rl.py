from copy import deepcopy
import gc
import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torchaudio
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import (
    default,
    exists,
    get_tokenizer,
    mask_from_start_end_indices,
)
from rl import reward


EPSILON=1e-6

def mask_from_frac_lengths(seq_len, frac_lengths):
    max_start = (frac_lengths * seq_len).long()

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    start = torch.min(start, dim=-1, keepdim=True).values.repeat(start.size(0))
    prompt_idx = mask_from_start_end_indices(seq_len, (0 * start).long(), start)
    trg_idx = mask_from_start_end_indices(seq_len, start, seq_len)

    return prompt_idx, trg_idx


class GRPOTrainer:
    def __init__(
        self,
        model: CFM,
        total_updates,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        min_samples=0,
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        logger: str | None = "wandb",  # "wandb" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        model_cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "total_updates": total_updates,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "min_samples": min_samples,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        self.model = model
        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.ref_model = deepcopy(self.model)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

        self.total_updates = total_updates
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5r-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.noise_scheduler = noise_scheduler

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.ref_model = self.accelerator.prepare(self.ref_model)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        checkpoint = torch.load(
            f"{self.checkpoint_path}/{latest_checkpoint}",
            weights_only=True,
            map_location="cpu",
        )

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint["model_state_dict"]
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(vocoder_name="vocos")
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=False if train_dataset.streaming else True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                min_samples=self.min_samples,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=True,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = self.total_updates
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        if train_dataset.streaming:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        else:
            train_dataloader, self.scheduler = self.accelerator.prepare(
                train_dataloader, self.scheduler
            )
        start_update = self.load_checkpoint()
        global_update = start_update

        for epoch in range(100):
            self.model.train()

            progress_bar = tqdm(
                range(total_updates),
                desc=f"Update {global_update}/{total_updates}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=0,
            )

            for batch in train_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)  # [b, t, d]
                    if mel_spec.shape[0] == 1:
                        print(f"skipping batch because mel_spec.shape[0] == 1")
                        continue
                    mel_lengths = batch["mel_lengths"]
                    if train_dataset.streaming:
                        mel_spec = mel_spec.to(self.accelerator.device)
                        mel_lengths = mel_lengths.to(self.accelerator.device)

                    text_len = max([len(item) for item in text_inputs])
                    if text_len > max(mel_lengths):
                        continue

                    process_index = self.accelerator.process_index
                    random_seed = random.randint(1, 50) + process_index
                    torch.manual_seed(random_seed)

                    frac_lengths = torch.zeros((mel_lengths.size(0),), device=mel_lengths.device)
                    frac_lengths = frac_lengths.float().uniform_(*(0.1, 0.3))
                    prompt_idx, trg_idx = mask_from_frac_lengths(
                        mel_lengths, frac_lengths
                    )
                    prompt_idx = prompt_idx.unsqueeze(-1)
                    prompt_idx = prompt_idx.repeat(1, 1, 100)
                    prompt_mel_spec = mel_spec[prompt_idx].view(
                        mel_spec.size(0), -1, mel_spec.size(2)
                    )
                    out, _, pro_result = self.model.forward_rl(
                        cond=prompt_mel_spec,
                        text=text_inputs,
                        duration=mel_lengths,
                        steps=nfe_step,
                        cfg_strength=2.0,
                        sway_sampling_coef=-1.0,
                    )
                    with torch.no_grad():
                        _, _, ref_pro_result = self.ref_model.forward_rl(
                            cond=prompt_mel_spec,
                            text=text_inputs,
                            duration=mel_lengths,
                            steps=nfe_step,
                            cfg_strength=2.0,
                            sway_sampling_coef=-1.0,
                        )
                    pro_result_sample = []
                    ref_pro_result_sample = []
                    for i, item in enumerate(pro_result):
                        if item[-1]:
                            pro_result_sample.append(item[:-1])
                            ref_pro_result_sample.append(ref_pro_result[i][:-1])
                    pro_result = pro_result_sample
                    ref_pro_result = ref_pro_result_sample
                    sim, acc = reward.get_reward(out, mel_spec)

                    rewards = sim * 1.0 + acc * 3.0

                    if self.accelerator.num_processes > 1 and dist.is_initialized():
                        rewards_flat = rewards.view(-1)

                        world_size = self.accelerator.num_processes
                        gathered_rewards = [
                            torch.zeros_like(rewards_flat) for _ in range(world_size)
                        ]
                        dist.all_gather(gathered_rewards, rewards_flat)
                        all_rewards = torch.stack(gathered_rewards, dim=0)
                        mean = all_rewards.mean(dim=0).mean()  # Global mean
                        std = all_rewards.std(dim=0).mean()  # Global std
                    else:
                        mean = rewards.mean()
                        std = rewards.std()

                    advantages = (rewards - mean) / (std + EPSILON)
                    advantages = advantages.detach()

                    pro_advantages = []
                    for x, mu, log_sigma in pro_result:
                        p = torch.exp(
                            -F.mse_loss(mu, x, reduction="none")
                            / (2 * (torch.exp(log_sigma) ** 2))
                        )
                        p = p / torch.exp(log_sigma)
                        pro_advantages.append(p)
                    pro_advantages = torch.stack(
                        pro_advantages, dim=1
                    )  # [batch, num_steps, seq_len, mel_dim]
                    advantages = advantages.view(advantages.size(0), 1, 1, 1)
                    pro_advantages = pro_advantages * advantages

                    # Reshape trg_idx to match pro_advantages dimensions
                    trg_idx = trg_idx.unsqueeze(-1).unsqueeze(1)
                    trg_idx = trg_idx.expand(
                        -1, pro_advantages.size(1), -1, pro_advantages.size(-1)
                    )
                    # pro_advantages = pro_advantages[trg_idx]
                    pro_advantages = torch.where(trg_idx, pro_advantages, torch.zeros_like(pro_advantages))
                    pro_advantages = pro_advantages.mean()

                    loss_kl = reward.get_kl(pro_result, ref_pro_result)
                    loss_kl = loss_kl.mean()

                    loss = - pro_advantages + loss_kl  # eq 10 in F5R-TTS paper
                    if torch.isnan(loss):
                        print(f"NaN loss detected at update {global_update}. Stopping training.")
                        self.save_checkpoint(global_update, last=True)
                        self.accelerator.end_training()
                        return

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    print(f"global_update: {global_update}, loss: {loss.item()}, pro_advantages: {pro_advantages.item()}, loss_kl: {loss_kl.item()}")
                    self.accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                            "loss_kl": loss_kl.item(),
                            "pro_advantages": pro_advantages.item(),
                        },
                        step=global_update,
                    )

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        sample_mel_length = mel_lengths[0]
                        ref_audio_len = sample_mel_length // 2
                        infer_text = [text_inputs[0]]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len, :].unsqueeze(0),
                                text=infer_text,
                                duration=sample_mel_length,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.permute(0, 2, 1)  # the second half is generated
                            reference = mel_spec[0][:sample_mel_length, :].unsqueeze(0).permute(0, 2, 1)
                            gen_mel_spec = generated.to(self.accelerator.device)
                            ref_mel_spec = reference.to(self.accelerator.device)
                            gen_audio = vocoder.decode(gen_mel_spec).cpu()
                            ref_audio = vocoder.decode(ref_mel_spec).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

                if global_update > total_updates:
                    break

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
