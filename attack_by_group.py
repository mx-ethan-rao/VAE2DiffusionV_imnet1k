#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack_grouped.py

Load a precomputed pullback npz (e.g. imnetv1_3k_pullback.npz), split member/heldout
into groups (random / median / quartiles), and run the TRUE-label probe (from attack.py)
on each group independently.

Usage example:
  python attack_grouped.py \
    --data_root /path/to/imagenet \
    --out_dir runs/ldm_imnet256_sdvae \
    --pullback_npz runs/ldm_imnet256/imnetv1_3k_pullback.npz \
    --grouping quartiles \
    --batch_size 128 \
    --probe_min_t 0 --probe_max_t 300 --probe_step 10
"""
from __future__ import annotations
import os, argparse, math
from typing import Tuple, Optional, List
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

# reuse the autoencoder and unet models loaded by attack.py (diffusers)
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchmetrics.classification import BinaryAUROC, BinaryROC


# ---------------------------
# small helpers copied/adapted from attack.py
# ---------------------------
def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

@torch.no_grad()
def vae_mean_latents(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    return vae.encode(x).latent_dist.mean

def unwrap_any(m):
    if hasattr(m, "module"): return m.module
    if hasattr(m, "_orig_mod"): return m._orig_mod
    return m

# copy probe function logic (adapted slightly to accept precomputed latents)
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

    auc_mtr, roc_mtr = BinaryAUROC().to(device), BinaryROC().to(device)
    auroc_k, tpr1_k, asr_k = [], [], []


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
        sm, sn = sm / scale, sn / scale
        scores = torch.cat([sm, sn])
        labels = torch.cat([torch.zeros_like(sm), torch.ones_like(sn)]).long()

        auroc = auc_mtr(scores, labels).item()
        fpr, tpr, _ = roc_mtr(scores, labels)
        # handle potentially empty mask
        idx = (fpr < 0.01).sum() - 1
        idx = max(int(idx.item() if torch.is_tensor(idx) else idx), 0)
        tpr_at1 = tpr[idx].item()
        asr = ((tpr + 1 - fpr) / 2).max().item()

        auroc_k.append(auroc)
        tpr1_k.append(tpr_at1)
        asr_k.append(asr)
        auc_mtr.reset(); roc_mtr.reset()

    print(f"AUROC  per‑step : {auroc_k}")
    print(f"TPR@1% per‑step : {tpr1_k}")
    print(f"ASR     per‑step: {asr_k}")
    print("\nBest over K steps")
    print(f"  AUROC  = {max(auroc_k):.4f}")
    print(f"  ASR    = {max(asr_k):.4f}")
    print(f"  TPR@1% = {max(tpr1_k):.4f}")

# ---------------------------
# grouping helper (robust to key names)
# ---------------------------
def load_pullback_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # possible keys based on your scripts: try multiple names
    def pick(keys):
        for k in keys:
            if k in data:
                return data[k]
        return None
    m_idx = pick(['member_indices', 'members_indices', 'member_idx', 'member_index'])
    h_idx = pick(['heldout_indices', 'held_indices', 'heldout_idx', 'heldout_index'])
    m_metric = pick(['member_metric', 'member_logvols', 'member_logvol', 'member_metrics', 'member_logvol'])
    h_metric = pick(['heldout_metric', 'heldout_logvols', 'heldout_logvol', 'heldout_metrics', 'heldout_logvol'])
    if m_idx is None or h_idx is None:
        raise KeyError(f"Couldn't find member/heldout index arrays in {npz_path}. Keys found: {list(data.keys())}")
    if m_metric is None or h_metric is None:
        # fall back to computing dummy metrics (equal) so random grouping still works
        print("[warn] pullback metrics not found; grouping will only support 'random' reliably.")
        m_metric = np.zeros_like(m_idx, dtype=np.float64)
        h_metric = np.zeros_like(h_idx, dtype=np.float64)
    # ensure numpy arrays
    return {
        'member_indices': np.array(m_idx),
        'heldout_indices': np.array(h_idx),
        'member_metric': np.array(m_metric).astype(np.float64),
        'heldout_metric': np.array(h_metric).astype(np.float64),
    }

def build_groups_from_npz(npz_path: str, grouping: str, seed: int = 2025, n_random_groups: int = 4):
    d = load_pullback_npz(npz_path)
    m_idx, h_idx = d['member_indices'], d['heldout_indices']
    m_mv, h_mv = d['member_metric'], d['heldout_metric']

    if grouping == 'random':
        rng = np.random.RandomState(seed)
        # create n_random_groups equal-ish splits
        m_perm = rng.permutation(len(m_idx))
        h_perm = rng.permutation(len(h_idx))
        m_groups = [m_idx[m_perm[i::n_random_groups]] for i in range(n_random_groups)]
        h_groups = [h_idx[h_perm[i::n_random_groups]] for i in range(n_random_groups)]
        names = [f"rand{i+1}" for i in range(n_random_groups)]
        print(f"[random] split into {n_random_groups} groups with seed={seed}")
    else:
        union = np.concatenate([m_mv, h_mv], axis=0)
        if grouping == 'median':
            thr = np.median(union)
            names = ['low','high']
            m_groups = [m_idx[m_mv <= thr], m_idx[m_mv > thr]]
            h_groups = [h_idx[h_mv <= thr], h_idx[h_mv > thr]]
            print(f"[threshold] median={thr:.6f}")
        elif grouping == 'quartiles':
            q1, med, q3 = np.percentile(union, [25, 50, 75])
            names = ['q1','q2','q3','q4']
            m_groups = [
                m_idx[m_mv <= q1],
                m_idx[(m_mv > q1) & (m_mv <= med)],
                m_idx[(m_mv > med) & (m_mv <= q3)],
                m_idx[m_mv > q3],
            ]
            h_groups = [
                h_idx[h_mv <= q1],
                h_idx[(h_mv > q1) & (h_mv <= med)],
                h_idx[(h_mv > med) & (h_mv <= q3)],
                h_idx[h_mv > q3],
            ]
            print(f"[thresholds] q1={q1:.6f}  med={med:.6f}  q3={q3:.6f}")
        else:
            raise ValueError("grouping must be 'random'|'median'|'quartiles'")
    for n, mg, hg in zip(names, m_groups, h_groups):
        print(f"Group {n:>6}: members={len(mg)} | heldout={len(hg)}")
    return dict(names=names, member_groups=m_groups, held_groups=h_groups)

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True, help="used to find default checkpoint if --ckpt not given")
    ap.add_argument("--pullback_npz", type=str, required=True)
    ap.add_argument("--vae_repo", type=str, default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--latent_ch", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--probe_min_t", type=int, default=0)
    ap.add_argument("--probe_max_t", type=int, default=300)
    ap.add_argument("--probe_step", type=int, default=10)
    ap.add_argument("--grouping", type=str, choices=['random','median','quartiles'], default='quartiles')
    ap.add_argument("--random_groups", type=int, default=2)
    ap.add_argument("--ckpt", type=str, default=None, help="explicit unet checkpoint path")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- dataset
    tf = build_transforms(args.img_size)
    train_full = ImageFolder(os.path.join(args.data_root, "train"), transform=tf)
    val_full   = ImageFolder(os.path.join(args.data_root, "val"),   transform=tf)

    # --- models (reuse attack.py approach) ---
    SCALE = 0.18215
    vae_dtype = torch.float16 if args.amp else torch.float32
    model_dtype = torch.float16 if args.amp else torch.float32

    print("[Model] loading SD VAE ...")
    vae = AutoencoderKL.from_pretrained(args.vae_repo, torch_dtype=vae_dtype)
    vae.eval().requires_grad_(False).to(device)

    print("[Model] building UNet ...")
    unet = UNet2DConditionModel(
        sample_size=args.img_size//8,
        in_channels=args.latent_ch,
        out_channels=args.latent_ch,
        layers_per_block=2,
        block_out_channels=(192, 384, 576, 960),
        down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"),
        cross_attention_dim=512,
        attention_head_dim=8,
    ).to(device)

    ckpt_path = args.ckpt if args.ckpt is not None else os.path.join(args.out_dir, "unet_last.pt")
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        unet.load_state_dict(ck["model"], strict=True)
    else:
        unet.load_state_dict(ck, strict=True)

    num_classes = int(ck.get("num_classes", 1000)) if isinstance(ck, dict) else 1000
    class_embed = torch.nn.Embedding(num_classes, 512).to(device)
    if isinstance(ck, dict) and "class_embed" in ck:
        try:
            class_embed.load_state_dict(ck["class_embed"])
        except Exception as e:
            print(f"[warn] failed to load class_embed: {e} (using fresh init)")

    unet = unet.to(dtype=model_dtype)
    class_embed = class_embed.to(dtype=model_dtype)
    unet.eval(); class_embed.eval()

    # build groups
    splits = build_groups_from_npz(args.pullback_npz, args.grouping, seed=args.seed, n_random_groups=args.random_groups)

    # For each group: build loaders, encode latents, run probe_unet
    for name, mem_idx, held_idx in zip(splits['names'], splits['member_groups'], splits['held_groups']):
        if len(mem_idx) == 0 or len(held_idx) == 0:
            print(f"\n=== Group {name} (skipped; empty) ===")
            continue

        print(f"\n=== Group {name} ===")
        m_loader = DataLoader(Subset(train_full, mem_idx.tolist()),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
        h_loader = DataLoader(Subset(val_full, held_idx.tolist()),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

        # collect latents and labels
        Zm_list, Lm_list = [], []
        for imgs, labs in tqdm(m_loader, desc=f"encode members [{name}]"):
            imgs = imgs.to(device)
            if vae_dtype == torch.float16: imgs = imgs.half()
            with torch.no_grad():
                z = vae_mean_latents(vae, imgs) * SCALE
            Zm_list.append(z.detach().to(device=device, dtype=model_dtype))
            Lm_list.append(labs)
        Zm = torch.cat(Zm_list, dim=0)
        Lm = torch.cat(Lm_list, dim=0).to(device)

        Zh_list, Lh_list = [], []
        for imgs, labs in tqdm(h_loader, desc=f"encode heldout [{name}]"):
            imgs = imgs.to(device)
            if vae_dtype == torch.float16: imgs = imgs.half()
            with torch.no_grad():
                z = vae_mean_latents(vae, imgs) * SCALE
            Zh_list.append(z.detach().to(device=device, dtype=model_dtype))
            Lh_list.append(labs)
        Zh = torch.cat(Zh_list, dim=0)
        Lh = torch.cat(Lh_list, dim=0).to(device)

        # run probe
        probe_unet(
            unet=unet, class_embed=class_embed,
            lat_member=Zm, labs_member=Lm,
            lat_nonmember=Zh, labs_nonmember=Lh,
            device=device,
            probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
        )

if __name__ == "__main__":
    main()
