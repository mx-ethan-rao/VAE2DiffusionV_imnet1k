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

# original probe_unet (kept for reference) - still available if you want to use it elsewhere.
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

    print(f"AUROC  per-step : {auroc_k}")
    print(f"TPR@1% per-step : {tpr1_k}")
    print(f"ASR     per-step: {asr_k}")
    print("\nBest over K steps")
    print(f"  AUROC  = {max(auroc_k):.4f}")
    print(f"  ASR    = {max(asr_k):.4f}")
    print(f"  TPR@1% = {max(tpr1_k):.4f}")

# ---------------------------
# streaming probe (encode a batch, attack batch, accumulate scalar scores)
# ---------------------------
@torch.no_grad()
def probe_unet_streaming(
    unet: UNet2DConditionModel,
    class_embed: torch.nn.Embedding,
    vae: AutoencoderKL,
    member_loader: DataLoader,
    heldout_loader: DataLoader,
    device: torch.device,
    probe_min_t: int, probe_max_t: int, probe_step: int,
    SCALE: float = 0.18215,
    vae_dtype=torch.float32,
    model_dtype=torch.float32,
):
    """
    Encode-a-batch-then-attack streaming mode:
      - iterate member_loader and heldout_loader batches
      - encode each batch with VAE mean, multiply by SCALE
      - for each probe t, compute unet predictions for that batch and collect scalar scores
      - at the end, concatenate all scores and compute metrics (AUROC, TPR@1%, ASR)
    """

    unet.eval(); class_embed.eval(); vae.eval()
    probe_ts = list(range(probe_min_t, probe_max_t, probe_step))
    print(f"[Probe-stream] timesteps: {probe_ts}")

    # containers: dict tval -> list of numpy floats
    sm_store = {t: [] for t in probe_ts}  # members
    sn_store = {t: [] for t in probe_ts}  # heldout
    labs_m_all = []  # not strictly needed, but keep parity if ever needed
    labs_n_all = []

    # helper to process a loader and fill store dict
    def process_loader(loader, store_dict, is_member: bool):
        for imgs, labs in tqdm(loader, desc=f"stream encode {'members' if is_member else 'heldout'}"):
            # move imgs to device and correct dtype for VAE
            imgs = imgs.to(device)
            if vae_dtype == torch.float16:
                imgs = imgs.half()
            with torch.no_grad():
                z = vae_mean_latents(vae, imgs) * SCALE  # latent on device
            # ensure dtype for model/unet
            z = z.to(device=device, dtype=model_dtype)

            # move labels to device and to correct dtype for class_embed lookup
            labs_dev = labs.to(device)

            # compute encoder_hidden_states once per batch (class_embed)
            # NOTE: class_embed might be float16/float32 depending on model_dtype
            ctx = class_embed(labs_dev).unsqueeze(1)  # (B,1,512)
            if model_dtype == torch.float16:
                ctx = ctx.half()
            # for each probe t compute predictions and scalar scores
            for tval in probe_ts:
                t_tensor = torch.full((z.size(0),), tval, device=device, dtype=torch.long)
                # UNet forward (we only need .sample)
                pred = unet(z, t_tensor, encoder_hidden_states=ctx).sample
                # per-sample scalar score as in original: sum(abs(pred)^4) over flatten dims
                s = (pred.abs() ** 4).flatten(1).sum(dim=-1)
                # move to cpu numpy floats to avoid holding tensors on GPU
                s_np = s.detach().cpu().float().numpy()
                store_dict[tval].append(s_np)
            # optionally keep labels (not strictly necessary)
            if is_member:
                labs_m_all.append(labs.numpy())
            else:
                labs_n_all.append(labs.numpy())

    # process members then heldout
    process_loader(member_loader, sm_store, is_member=True)
    process_loader(heldout_loader, sn_store, is_member=False)

    # now for each t, concatenate scores and compute metrics
    device_cpu = device  # metrics will be computed on device (GPU) if available
    auc_mtr = BinaryAUROC().to(device_cpu)
    roc_mtr = BinaryROC().to(device_cpu)

    auroc_k, tpr1_k, asr_k = [], [], []

    for tval in probe_ts:
        # concatenate stored numpy arrays
        sm_all = np.concatenate(sm_store[tval], axis=0) if len(sm_store[tval]) > 0 else np.zeros(0, dtype=np.float32)
        sn_all = np.concatenate(sn_store[tval], axis=0) if len(sn_store[tval]) > 0 else np.zeros(0, dtype=np.float32)

        if sm_all.size == 0 or sn_all.size == 0:
            print(f"[warn] empty member/heldout for t={tval}; skipping metric computation for this t")
            auroc_k.append(float("nan")); tpr1_k.append(float("nan")); asr_k.append(float("nan"))
            continue

        # scale normalization same as original: divide by max(max(sm), max(sn), 1e-12)
        scale = max(float(sm_all.max()), float(sn_all.max()), 1e-12)
        sm_all = sm_all / scale
        sn_all = sn_all / scale

        # build torch tensors for metrics on device
        scores = torch.tensor(np.concatenate([sm_all, sn_all], axis=0), device=device_cpu, dtype=torch.float32)
        labels = torch.cat([
            torch.zeros(sm_all.shape[0], dtype=torch.long, device=device_cpu),
            torch.ones(sn_all.shape[0], dtype=torch.long, device=device_cpu),
        ])

        # compute AUROC and ROC curve
        auroc = float(auc_mtr(scores, labels).item())
        fpr, tpr, _ = roc_mtr(scores, labels)
        # handle potentially empty mask
        idx = (fpr < 0.01).sum() - 1
        idx = max(int(idx.item() if torch.is_tensor(idx) else idx), 0)
        tpr_at1 = float(tpr[idx].item())
        asr = float(((tpr + 1 - fpr) / 2).max().item())

        auroc_k.append(auroc)
        tpr1_k.append(tpr_at1)
        asr_k.append(asr)
        auc_mtr.reset(); roc_mtr.reset()

    print(f"AUROC  per-step : {auroc_k}")
    print(f"TPR@1% per-step : {tpr1_k}")
    print(f"ASR     per-step: {asr_k}")
    print("\nBest over K steps")
    # guard against all-nan
    finite_aucs = [x for x in auroc_k if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    finite_asrs = [x for x in asr_k if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    finite_tprs = [x for x in tpr1_k if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if len(finite_aucs) > 0:
        print(f"  AUROC  = {max(finite_aucs):.4f}")
    else:
        print("  AUROC  = nan")
    if len(finite_asrs) > 0:
        print(f"  ASR    = {max(finite_asrs):.4f}")
    else:
        print("  ASR    = nan")
    if len(finite_tprs) > 0:
        print(f"  TPR@1% = {max(finite_tprs):.4f}")
    else:
        print("  TPR@1% = nan")


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

    # For each group: build loaders, encode latents, run probe_unet (streaming version)
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

        # STREAMING probe: encode a batch, attack batch, accumulate scores (memory-friendly)
        probe_unet_streaming(
            unet=unet, class_embed=class_embed, vae=vae,
            member_loader=m_loader, heldout_loader=h_loader,
            device=device,
            probe_min_t=args.probe_min_t, probe_max_t=args.probe_max_t, probe_step=args.probe_step,
            SCALE=SCALE, vae_dtype=vae_dtype, model_dtype=model_dtype,
        )

if __name__ == "__main__":
    main()
