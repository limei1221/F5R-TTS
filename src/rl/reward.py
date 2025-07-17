import torch

from f5_tts.infer.utils_infer import load_vocoder
from rl import utils


vocos = load_vocoder(vocoder_name="vocos")


def get_reward(gen_mel, trg_mel):
    gen_mel = gen_mel.permute(0, 2, 1)
    trg_mel = trg_mel.permute(0, 2, 1)
    # Move mel spectrograms to the same device as vocoder
    vocoder_device = next(vocos.parameters()).device
    gen_mel = gen_mel.to(vocoder_device)
    trg_mel = trg_mel.to(vocoder_device)
    gen_wav = vocos.decode(gen_mel)
    trg_wav = vocos.decode(trg_mel)

    gen_emb = utils.get_emb(gen_wav, 24000)
    trg_emb = utils.get_emb(trg_wav, 24000)
    sim = utils.cal_sim(gen_emb, trg_emb).to(vocoder_device)

    gen_txt = utils.get_asr(gen_wav, 24000)
    trg_txt = utils.get_asr(trg_wav, 24000)
    acc = []
    for r, h in zip(trg_txt, gen_txt):
        acc.append(1 - utils.cal_wer(r, h))
    acc = torch.tensor(acc).to(vocoder_device)
    return sim, acc


def cal_kl(gen, ref):
    gen_mu, gen_log_sigma = gen
    ref_mu, ref_log_sigma = ref

    # KL divergence between two Gaussian distributions
    # KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

    # σ² = exp(2 * logσ)
    gen_var = torch.exp(2 * gen_log_sigma)
    ref_var = torch.exp(2 * ref_log_sigma)

    kl = ref_log_sigma - gen_log_sigma
    kl += (gen_var + (gen_mu - ref_mu) ** 2) / (2 * ref_var)
    # kl -= 0.5

    return kl


def get_kl(gen_pros, ref_pros):
    loss = 0
    for gen, ref in zip(gen_pros, ref_pros):
        loss += cal_kl(gen[1: 3], ref[1: 3])
    return loss
