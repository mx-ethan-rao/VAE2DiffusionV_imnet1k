#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
attack.py — probe the class-conditional LDM trained by main.py using TRUE LABELS

- Loads the same SD VAE as training (AutoencoderKL.from_pretrained) with SCALE=0.18215.
- Builds UNet2DConditionModel exactly like main.py and loads {out_dir}/unet_last.pt.
- Loads class_embed + num_classes from checkpoint (or infers num_classes).
- Encodes train (members) and val (non-members) images to latents (mean * SCALE).
- Feeds the **true labels** to class_embed for conditioning at probe time.
- For t in [probe_min_t, probe_max_t) with step probe_step:
    score(x) = sum(|unet(z_t, t, ctx)|^4)  (per sample)
  Computes AUROC, TPR@FPR<1%, ASR; prints the best over t.

Run:
  python attack.py --data_root /path/to/imagenet --out_dir runs/ldm_imnet256_sdvae \
                   --batch_size 128 --probe_min_t 0 --probe_max_t 300 --probe_step 10
"""

from __future__ import annotations
import os, argparse
from typing import Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel


# ---------------------------
# Data / utils
# ---------------------------
def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def sample_per_class(imfolder: ImageFolder, n_per_class: int, seed: int):
    rng = __import__("random").Random(seed)
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

def unwrap_any(m):
    if hasattr(m, "module"): return m.module
    if hasattr(m, "_orig_mod"): return m._orig_mod
    return m

# ---------------------------
# Simple ROC/AUC on numpy
# ---------------------------
def roc_auc_and_curve(scores: torch.Tensor, labels: torch.Tensor):
    s = scores.detach().cpu().numpy()
    l = labels.detach().cpu().numpy()
    order = np.argsort(-s)
    s_sorted, l_sorted = s[order], l[order]
    P = float((l == 1).sum()); N = float((l == 0).sum())
    if P == 0 or N == 0:
        return float("nan"), torch.tensor([]), torch.tensor([]), torch.tensor([])
    tprs, fprs, thresholds = [], [], []
    tp = fp = 0.0
    prev_s = None
    for i in range(len(s_sorted)):
        if prev_s is None or s_sorted[i] != prev_s:
            tprs.append(tp / P); fprs.append(fp / N); thresholds.append(s_sorted[i]); prev_s = s_sorted[i]
        if l_sorted[i] == 1: tp += 1.0
        else: fp += 1.0
    tprs.append(tp / P); fprs.append(fp / N); thresholds.append(s_sorted[-1] - 1e-12)
    fprs_arr, tprs_arr = np.array(fprs), np.array(tprs)
    sort2 = np.argsort(fprs_arr)
    fprs_arr, tprs_arr = fprs_arr[sort2], tprs_arr[sort2]
    auc = np.trapz(tprs_arr, fprs_arr)
    return float(auc), torch.from_numpy(fprs_arr), torch.from_numpy(tprs_arr), torch.from_numpy(np.array(thresholds)[sort2])

# ---------------------------
# VAE helper
# ---------------------------
@torch.no_grad()
def vae_mean_latents(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    return vae.encode(x).latent_dist.mean

# ---------------------------
# Probe using TRUE labels (AMP-safe)
# ---------------------------
@torch.no_grad()
def probe_unet(
    unet: UNet2DConditionModel,
    class_embed: torch.nn.Embedding,
    lat_member: torch.Tensor, labs_member: torch.Tensor,
    lat_nonmember: torch.Tensor, labs_nonmember: torch.Tensor,
    device: torch.device,
    probe_min_t: int, probe_max_t: int, probe_step: int,
):
    unet.eval(); class_embed.eval()
    N_m, N_n = lat_member.size(0), lat_nonmember.size(0)

    probe_ts = list(range(probe_min_t, probe_max_t, probe_step))
    best = {"t": None, "auroc": -1.0, "tpr1": 0.0, "asr": 0.0}
    print(f"[Probe] timesteps: {probe_ts}")

    # Pre-move labels to device
    labs_member = labs_member.to(device)
    labs_nonmember = labs_nonmember.to(device)

    for tval in tqdm(probe_ts):
        t_m = torch.full((N_m,), tval, device=device, dtype=torch.long)
        t_n = torch.full((N_n,), tval, device=device, dtype=torch.long)

        # TRUE label conditioning (inherits dtype from class_embed's weight)
        ctx_m = class_embed(labs_member).unsqueeze(1)     # (N_m, 1, 512)
        ctx_n = class_embed(labs_nonmember).unsqueeze(1)  # (N_n, 1, 512)

        pred_m = unet(lat_member, t_m, encoder_hidden_states=ctx_m).sample
        pred_n = unet(lat_nonmember, t_n, encoder_hidden_states=ctx_n).sample

        sm = (pred_m.abs() ** 4).flatten(1).sum(dim=-1)
        sn = (pred_n.abs() ** 4).flatten(1).sum(dim=-1)

        scale = max(float(sm.max().detach().cpu().item()), float(sn.max().detach().cpu().item()), 1e-12)
        sm, sn = sm/scale, sn/scale

        scores = torch.cat([sm, sn], 0)
        labels = torch.cat([torch.zeros_like(sm), torch.ones_like(sn)], 0).long()

        auc, fpr, tpr, _ = roc_auc_and_curve(scores, labels)
        if np.isnan(auc):
            print("[warn] AUC undefined (missing class) — skipping this t")
            continue

        fpr_np, tpr_np = fpr.numpy(), tpr.numpy()
        mask = (fpr_np < 0.01)
        tpr_at1 = float(tpr_np[mask.argmax()]) if mask.any() else 0.0
        asr = float(((tpr_np + 1.0 - fpr_np) / 2.0).max())

        print(f"t={tval:4d}  AUROC={auc:.4f}  TPR@FPR<1%={tpr_at1:.4f}  ASR={asr:.4f}")

        if auc > best["auroc"]:
            best.update({"t": tval, "auroc": auc, "tpr1": tpr_at1, "asr": asr})

    print("\n[Best over timesteps]")
    print(f"  t = {best['t']}")
    print(f"  AUROC      = {best['auroc']:.4f}")
    print(f"  TPR@FPR<1% = {best['tpr1']:.4f}")
    print(f"  ASR        = {best['asr']:.4f}")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Attack (TRUE labels) for class-conditional LDM trained by main.py")
    ap.add_argument("--data_root", type=str, required=True, help="ImageNet root with train/ and val/")
    ap.add_argument("--out_dir", type=str, required=True, help="training out_dir (contains unet_last.pt)")
    ap.add_argument("--ckpt", type=str, default=None, help="optional explicit path to checkpoint")
    ap.add_argument("--vae_repo", type=str, default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--latent_ch", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--n_per_class_member", type=int, default=3)
    ap.add_argument("--n_per_class_nonmember", type=int, default=3)
    ap.add_argument("--probe_min_t", type=int, default=0)
    ap.add_argument("--probe_max_t", type=int, default=300)
    ap.add_argument("--probe_step", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # VAE encode dtype (matches training AMP behavior)
    vae_dtype = torch.float16 if args.amp else torch.float32
    # Model / inputs dtype for UNet + class_embed (must MATCH to avoid matmul dtype errors)
    model_dtype = torch.float16 if args.amp else torch.float32

    # --- data
    tf = build_transforms(args.img_size)
    train_full = ImageFolder(os.path.join(args.data_root, "train"), transform=tf)
    val_full   = ImageFolder(os.path.join(args.data_root, "val"),   transform=tf)
    # val_full   = ImageFolder('/banana/ethan/MIA_data/IMAGENETv2/ImageNetV2-matched-frequency',   transform=tf)

    mem_idx    = sample_per_class(train_full, args.n_per_class_member, seed=args.seed)
    nonmem_idx = sample_per_class(val_full,   args.n_per_class_nonmember, seed=args.seed)

    member_ds    = Subset(train_full, mem_idx)
    nonmember_ds = Subset(val_full,   nonmem_idx)

    member_loader = DataLoader(member_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    nonmember_loader = DataLoader(nonmember_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    # --- models
    SCALE = 0.18215
    print("[Model] loading SD VAE ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_repo, torch_dtype=vae_dtype)
    vae.eval().requires_grad_(False).to(device)

    unet = UNet2DConditionModel(
        sample_size=args.img_size//8,
        in_channels=args.latent_ch,
        out_channels=args.latent_ch,
        layers_per_block=2,
        block_out_channels=(192, 384, 576, 960),
        down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"),
        cross_attention_dim=512,
        attention_head_dim=8,  # matches main.py
    ).to(device)
    # load checkpoint
    ckpt_path = args.ckpt if args.ckpt is not None else os.path.join(args.out_dir, "unet_last.pt")
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        unet.load_state_dict(ck["model"], strict=True)
    else:
        unet.load_state_dict(ck, strict=True)

    # class embed
    num_classes = int(ck.get("num_classes", 1000)) if isinstance(ck, dict) else 1000
    class_embed = torch.nn.Embedding(num_classes, 512).to(device)
    if isinstance(ck, dict) and "class_embed" in ck:
        try:
            class_embed.load_state_dict(ck["class_embed"])
        except Exception as e:
            print(f"[warn] failed to load class_embed: {e} (using fresh init)", flush=True)

    # >>> Ensure UNet + class_embed dtypes match inputs <<<
    unet = unet.to(dtype=model_dtype)
    class_embed = class_embed.to(dtype=model_dtype)
    unet.eval(); class_embed.eval()

    # --- encode latents + collect TRUE labels
    @torch.no_grad()
    def encode_all(loader) -> Tuple[torch.Tensor, torch.Tensor]:
        zs, labs = [], []
        for x, y in tqdm(loader, desc="encoding"):
            x = x.to(device)
            # VAE runs in vae_dtype (fp16 if AMP)
            if vae_dtype == torch.float16:
                x = x.half()
            z_mu = vae.encode(x).latent_dist.mean * SCALE
            # store on CPU first to save VRAM
            zs.append(z_mu.detach().cpu()); labs.append(y)
        Z = torch.cat(zs, 0)
        L = torch.cat(labs, 0)
        # return latents on device with the **model dtype** so it matches UNet & class_embed
        return Z.to(device=device, dtype=model_dtype), L.to(device)

    print("[Encode] members ...")
    lat_m, labs_m = encode_all(member_loader)
    print("[Encode] non-members ...")
    lat_n, labs_n = encode_all(nonmember_loader)

    # --- probe with TRUE labels (everything already model_dtype) ---
    probe_unet(
        unet=unet, class_embed=class_embed,
        lat_member=lat_m, labs_member=labs_m,
        lat_nonmember=lat_n, labs_nonmember=labs_n,
        device=device,
        probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
    )

if __name__ == "__main__":
    main()
