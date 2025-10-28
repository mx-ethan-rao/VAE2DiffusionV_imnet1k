#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class-Conditional Latent Diffusion on ImageNet-1K (subset 100/class)
Using pretrained Stable Diffusion VAE (AutoencoderKL, frozen)
- Inputs:  3x256x256 in [-1,1]
- Latents: 4x32x32 (8x downsample) with SD latent scale 0.18215
- Conditioning: class label -> 512-d embedding -> UNet cross-attention
- Multi-GPU via 🤗 Accelerate

Resume:
  --resume loads UNet/class_embed/optimizer/scheduler/epoch from --resume_path
  (default: {out_dir}/unet_last.pt) and continues training at start_epoch.

Saves:
  - out_dir/train_indices_balanced.json
  - out_dir/vae_recon.png                     (quick recon sanity check)
  - out_dir/unet_last.pt                      (checkpoint with epoch)
  - out_dir/ldm_samples_ddim_e{epoch}.png     (periodic)
  - out_dir/ldm_samples_ddpm_final.png        (final)
  - out_dir/ldm_samples_ddim_final.png        (final)
"""
from __future__ import annotations
import os, json, random, argparse
from dataclasses import dataclass
from typing import Optional, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup


# ------------------------------
# Utils
# ------------------------------

def unwrap_any(model):
    m = model
    if hasattr(m, "module"): m = m.module
    if hasattr(m, "_orig_mod"): m = m._orig_mod
    return m

def save_grid_img(t: torch.Tensor, path: str, nrow=4, normalize=True, value_range=(-1,1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(t, nrow=nrow, normalize=normalize, value_range=value_range)
    save_image(grid, path)

def build_transforms(img_size: int, in_ch: int, aug: bool):
    t = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ]
    if aug:
        t.append(transforms.RandomHorizontalFlip())
    t += [transforms.ToTensor(), transforms.Normalize([0.5]*in_ch, [0.5]*in_ch)]
    return transforms.Compose(t)

def sample_per_class(imfolder: ImageFolder, n_per_class: int, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for idx, (_, lbl) in enumerate(imfolder.samples):
        buckets[lbl].append(idx)
    subset = []
    for lbl, inds in buckets.items():
        if len(inds) < n_per_class:
            raise RuntimeError(f"class {lbl} has only {len(inds)} images (<{n_per_class})")
        rng.shuffle(inds)
        subset.extend(inds[:n_per_class])
    rng.shuffle(subset)
    return subset

def save_indices(indices: List[int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(indices, f)

def load_indices(path: str) -> Optional[List[int]]:
    if not os.path.isfile(path): return None
    with open(path, "r") as f: return json.load(f)


# ------------------------------
# Diffusion schedule
# ------------------------------

class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=1.5e-3, beta_end=1.95e-2, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        ab = self.alpha_bars
        ab_prev = torch.cat([torch.tensor([1.0], device=device), ab[:-1]], dim=0)
        self.posterior_variance = betas * (1.0 - ab_prev) / (1.0 - ab).clamp(min=1e-12)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1,1,1,1)
        sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0(self, x_t, t, eps):
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1,1,1,1)
        sqrt_1mab = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1,1,1,1)
        return (x_t - sqrt_1mab*eps) / (sqrt_ab + 1e-8)

    def p_mean_variance(self, x_t, t, eps):
        ab_t = self.alpha_bars[t].view(-1,1,1,1)
        ab_prev = self.alpha_bars[(t-1).clamp(min=0)].view(-1,1,1,1)
        beta_t = self.betas[t].view(-1,1,1,1)
        alpha_t = self.alphas[t].view(-1,1,1,1)
        x0 = self.predict_x0(x_t, t, eps)
        coef1 = (torch.sqrt(ab_prev) * beta_t) / (1.0 - ab_t).clamp(min=1e-12)
        coef2 = (torch.sqrt(alpha_t) * (1.0 - ab_prev)) / (1.0 - ab_t).clamp(min=1e-12)
        mean = coef1 * x0 + coef2 * x_t
        var = self.posterior_variance[t].view(-1,1,1,1)
        logvar = self.posterior_log_variance_clipped[t].view(-1,1,1,1)
        return mean, var, logvar


# ------------------------------
# Config
# ------------------------------

@dataclass
class TrainCfg:
    data_root: str
    out_dir: str = "runs/ldm_imnet256_sdvae"
    img_size: int = 256
    latent_ch: int = 4

    # training/runtime
    per_device_batch: int = 32
    grad_accum: int = 1
    num_workers: int = 8
    prefetch_factor: int = 4
    n_train_per_class: int = 100
    seed: int = 2025
    amp: bool = False
    resume: bool = False
    heartbeat_s: float = 5.0

    # UNet / diffusion
    unet_base_lr: float = 1e-4
    unet_epochs: int = 350
    unet_warmup_steps: int = 1000
    T: int = 1000
    beta_start: float = 1.5e-3
    beta_end: float = 1.95e-2
    sample_every: int = 100
    save_every: int = 10


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="runs/ldm_imnet256_sdvae")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--per_device_batch", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--n_train_per_class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")  # optional explicit path
    parser.add_argument("--unet_epochs", type=int, default=350)
    parser.add_argument("--unet_base_lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1.5e-3)
    parser.add_argument("--beta_end", type=float, default=1.95e-2)
    parser.add_argument("--sample_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sd_vae_repo", default="stabilityai/sd-vae-ft-mse")
    args = parser.parse_args()

    proj = ProjectConfiguration(project_dir=args.out_dir, automatic_checkpoint_naming=True)
    accelerator = Accelerator(
        mixed_precision="fp16" if args.amp else "no",
        project_config=proj,
        gradient_accumulation_steps=args.grad_accum,
        log_with=None,
    )

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[Accelerate] processes={accelerator.num_processes}", flush=True)

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cfg = TrainCfg(
        data_root=args.data_root, out_dir=args.out_dir, img_size=args.img_size,
        per_device_batch=args.per_device_batch, grad_accum=args.grad_accum,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        n_train_per_class=args.n_train_per_class, seed=args.seed, amp=args.amp,
        resume=args.resume, unet_epochs=args.unet_epochs, unet_base_lr=args.unet_base_lr,
        T=args.T, beta_start=args.beta_start, beta_end=args.beta_end,
        sample_every=args.sample_every, save_every=args.save_every,
    )

    # ---------- Data ----------
    def build_loader(split):
        tf = build_transforms(cfg.img_size, 3, aug=(split=="train"))
        return ImageFolder(os.path.join(cfg.data_root, split), transform=tf)

    train_ds_full = build_loader("train")
    val_ds        = build_loader("val")
    num_classes = len(train_ds_full.classes)

    # Balanced subset indices
    idx_path = os.path.join(cfg.out_dir, "train_indices_balanced.json")
    if accelerator.is_main_process:
        saved = load_indices(idx_path)
        if saved is None:
            saved = sample_per_class(train_ds_full, cfg.n_train_per_class, cfg.seed)
            save_indices(saved, idx_path)
            print(f"[Subset] Sampled {len(saved)} indices (n/class={cfg.n_train_per_class})", flush=True)
        else:
            print(f"[Subset] Loaded {len(saved)} indices", flush=True)
    accelerator.wait_for_everyone()
    idx = load_indices(idx_path)
    train_ds = Subset(train_ds_full, idx)

    common_loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True
    )
    if cfg.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=cfg.per_device_batch, shuffle=True, **common_loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.per_device_batch, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=(cfg.num_workers>0), drop_last=False,
                              **({"prefetch_factor": cfg.prefetch_factor} if cfg.num_workers>0 else {}))

    # ---------- Pretrained SD VAE (frozen) ----------
    dtype = torch.float16 if args.amp else torch.float32
    vae = AutoencoderKL.from_pretrained(args.sd_vae_repo, torch_dtype=dtype)
    vae.eval().requires_grad_(False)
    vae = vae.to(accelerator.device)  # ensure weights on GPU
    SCALE = 0.18215

    # Sanity latent shape
    with torch.no_grad():
        dummy = torch.zeros(1,3,cfg.img_size,cfg.img_size, device=accelerator.device, dtype=dtype)
        posterior = vae.encode(dummy).latent_dist
        z = posterior.mean * SCALE
        assert z.shape[1:] == (cfg.latent_ch, cfg.img_size//8, cfg.img_size//8), f"latent shape mismatch: {z.shape}"

    # Quick recon grid
    if accelerator.is_main_process:
        with torch.no_grad():
            x_val, _ = next(iter(val_loader))
            x_vis = x_val[:32].to(accelerator.device)
            if args.amp: x_vis = x_vis.half()
            z = vae.encode(x_vis).latent_dist.mean * SCALE
            rec = vae.decode(z / SCALE).sample
            save_grid_img(torch.cat([x_vis.float().cpu(), rec.float().cpu()], 0),
                          os.path.join(cfg.out_dir, "vae_recon.png"), nrow=8)

    # ---------- UNet (conditional) ----------
    unet = UNet2DConditionModel(
        sample_size=cfg.img_size//8,  # 32
        in_channels=cfg.latent_ch,    # 4
        out_channels=cfg.latent_ch,   # predict noise
        layers_per_block=2,
        block_out_channels=(192, 384, 576, 960),
        down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"),
        cross_attention_dim=512,
        attention_head_dim=8,         # you can bump to 8 later
    )
    if args.grad_ckpt:
        unet.enable_gradient_checkpointing()
    if args.compile:
        try:
            unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            if accelerator.is_main_process: print(f"[warn] torch.compile(unet) failed: {e}", flush=True)

    class_embed = torch.nn.Embedding(num_classes, 512)

    diff = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=accelerator.device)
    opt = torch.optim.AdamW(list(unet.parameters()) + list(class_embed.parameters()),
                            lr=cfg.unet_base_lr, betas=(0.9,0.999), weight_decay=0.0)
    total_steps = max(1, (len(train_loader)//max(1,cfg.grad_accum)) * cfg.unet_epochs)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=cfg.unet_warmup_steps, num_training_steps=total_steps)

    # ---- Resume (load BEFORE prepare) ----
    start_epoch = 1
    ckpt_path_default = os.path.join(cfg.out_dir, "unet_last.pt")
    ckpt_to_load = args.resume_path if (args.resume_path and os.path.isfile(args.resume_path)) else ckpt_path_default
    if cfg.resume and os.path.isfile(ckpt_to_load):
        state = torch.load(ckpt_to_load, map_location="cpu")
        # load raw (unprepared) modules and optimizers/schedulers
        unet.load_state_dict(state["model"], strict=True)
        if "class_embed" in state:
            class_embed.load_state_dict(state["class_embed"])
        if "opt" in state:
            opt.load_state_dict(state["opt"])
        if "sched" in state:
            try:
                sched.load_state_dict(state["sched"])
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[warn] scheduler state load failed: {e}", flush=True)
        start_epoch = int(state.get("epoch", 0)) + 1
        if accelerator.is_main_process:
            print(f"[Resume] Loaded {ckpt_to_load}; start_epoch={start_epoch}", flush=True)
    elif cfg.resume:
        if accelerator.is_main_process:
            print(f"[Resume] No checkpoint found at {ckpt_to_load}; starting fresh.", flush=True)

    # ---- prepare for distributed ----
    unet, class_embed, opt, sched, train_loader_u = accelerator.prepare(
        unet, class_embed, opt, sched, train_loader
    )

    # ---------- Train loop ----------
    for epoch in range(start_epoch, cfg.unet_epochs + 1):
        unet.train(); class_embed.train()
        running = 0.0; count = 0
        pbar = tqdm(total=len(train_loader_u), disable=not accelerator.is_main_process,
                    ascii=True, desc=f"[LDM] e{epoch}/{cfg.unet_epochs}")

        for x, y in train_loader_u:
            x = x.to(accelerator.device, non_blocking=True)
            if args.amp: x = x.half()
            y = y.to(accelerator.device, non_blocking=True)

            # encode to latents (deterministic mean)
            with torch.no_grad(), accelerator.autocast():
                z0 = vae.encode(x).latent_dist.mean * SCALE  # (B,4,32,32)

            B = z0.size(0)
            t = torch.randint(0, diff.T, (B,), device=accelerator.device, dtype=torch.long)
            noise = torch.randn_like(z0)

            with accelerator.accumulate(unet), accelerator.autocast():
                zt = diff.q_sample(z0, t, noise)
                ctx = class_embed(y).unsqueeze(1)            # (B,1,512)
                pred = unet(zt, t, encoder_hidden_states=ctx).sample
                loss = F.mse_loss(pred, noise)

                accelerator.backward(loss)
                opt.step(); sched.step(); opt.zero_grad()

            bs = x.size(0)
            running += float(loss.detach()) * bs; count += bs
            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{running/max(1,count):.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
                pbar.update(1)

        if accelerator.is_main_process:
            pbar.close()
            # Save checkpoint
            if (epoch % cfg.save_every) == 0 or epoch == cfg.unet_epochs:
                torch.save(
                    {"model": unwrap_any(unet).state_dict(),
                     "class_embed": unwrap_any(class_embed).state_dict(),
                     "opt": opt.state_dict(),
                     "sched": sched.state_dict(),
                     "epoch": epoch,
                     "num_classes": num_classes},
                    ckpt_path_default,
                )
            # Periodic DDIM sampling
            if (epoch % cfg.sample_every) == 0 or epoch == cfg.unet_epochs:
                with torch.no_grad(), accelerator.autocast():
                    unwrap_any(unet).eval(); unwrap_any(class_embed).eval()
                    n = 16; steps = 50
                    C, H, W = cfg.latent_ch, cfg.img_size//8, cfg.img_size//8
                    x_t = torch.randn((n, C, H, W), device=accelerator.device)
                    y_samp = torch.randint(0, num_classes, (n,), device=accelerator.device)
                    ctx = unwrap_any(class_embed)(y_samp).unsqueeze(1)
                    ts = torch.linspace(cfg.T-1, 0, steps, device=accelerator.device).long()
                    for i in range(steps):
                        t_s = ts[i].repeat(n)
                        eps = unwrap_any(unet)(x_t, t_s, encoder_hidden_states=ctx).sample
                        x0 = diff.predict_x0(x_t, t_s, eps)
                        if i == steps-1:
                            x_t = x0; break
                        t_next = ts[i+1].repeat(n)
                        ab_next = diff.alpha_bars[t_next].view(-1,1,1,1)
                        x_t = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps
                    imgs_ddim = vae.decode(x_t / SCALE).sample.clamp(-1,1).cpu()
                    save_grid_img(imgs_ddim, os.path.join(cfg.out_dir, f"ldm_samples_ddim_e{epoch}.png"), nrow=4)

        accelerator.wait_for_everyone()

    # ---------- Final Sampling ----------
    if accelerator.is_main_process:
        unwrap_any(unet).eval(); unwrap_any(class_embed).eval()
        with torch.no_grad(), accelerator.autocast():
            n = 16; C, H, W = cfg.latent_ch, cfg.img_size//8, cfg.img_size//8
            y_samp = torch.randint(0, num_classes, (n,), device=accelerator.device)
            ctx = unwrap_any(class_embed)(y_samp).unsqueeze(1)

            # DDPM
            x = torch.randn((n, C, H, W), device=accelerator.device)
            for t_int in reversed(range(diff.T)):
                t = torch.full((n,), t_int, device=accelerator.device, dtype=torch.long)
                eps = unwrap_any(unet)(x, t, encoder_hidden_states=ctx).sample
                mean, var, logvar = diff.p_mean_variance(x, t, eps)
                if t_int > 0:
                    x = mean + torch.exp(0.5*logvar) * torch.randn_like(x)
                else:
                    x = mean
            imgs_ddpm = vae.decode(x / SCALE).sample.clamp(-1,1).cpu()
            save_grid_img(imgs_ddpm, os.path.join(cfg.out_dir, "ldm_samples_ddpm_final.png"), nrow=4)

            # DDIM
            steps = 50
            x = torch.randn((n, C, H, W), device=accelerator.device)
            ts = torch.linspace(cfg.T-1, 0, steps, device=accelerator.device).long()
            for i in range(steps):
                t = ts[i].repeat(n)
                eps = unwrap_any(unet)(x, t, encoder_hidden_states=ctx).sample
                x0 = diff.predict_x0(x, t, eps)
                if i == steps-1:
                    x = x0; break
                t_next = ts[i+1].repeat(n)
                ab_next = diff.alpha_bars[t_next].view(-1,1,1,1)
                x = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps
            imgs_ddim = vae.decode(x / SCALE).sample.clamp(-1,1).cpu()
            save_grid_img(imgs_ddim, os.path.join(cfg.out_dir, "ldm_samples_ddim_final.png"), nrow=4)

        print("[Done] Final samples saved (DDPM + DDIM).", flush=True)


if __name__ == "__main__":
    main()
