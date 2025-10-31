#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cal_pullback_metric.py  (per-dimension contributions)

Compute per-sample per-dimension pullback metric contributions for the SD VAE decoder:
  - z(x) = μ(x) * SCALE, where μ is VAE encoder mean, SCALE=0.18215
  - decoder mapping g(z) = decode(z / SCALE).sample  (pixel space in [-1, 1])
  - Jacobian J(z) = ∂g/∂z  (flattened output wrt flattened latent)

Per-dimension contributions (coordinate basis):
  c_k(z) = 0.5 * log( (J(z)^T J(z))_{kk} + eps )

We estimate diag(J^T J) via a Hutchinson estimator:
  diag(J^T J) = E_v[ (J^T v) ⊙ (J^T v) ],  v ~ N(0, I)
and average over n_mc probes.

Outputs (NPZ):
  - member_indices                 : indices used from ImageFolder(train)
  - member_per_dim_logcontrib      : (N_member, C*H*W)
  - heldout_indices                : indices used from ImageFolder(val)
  - heldout_per_dim_logcontrib     : (N_heldout, C*H*W)
  - meta                           : dict with config

Example:
  python cal_pullback_metric.py \
    --data_root /path/imagenet \
    --out_dir runs/ldm_imnet256_sdvae \
    --out_npz runs/ldm_imnet256_sdvae/pullback_perdim.npz \
    --sd_vae_repo stabilityai/sd-vae-ft-mse \
    --img_size 256 --encode_batch 64 \
    --n_mc 64 --eps 1e-12
"""

import os, json, argparse, random
from typing import List, Optional, Dict
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Subset

# ---------- Utils (mirrors main.py where relevant) ----------
def build_transforms(img_size: int, in_ch: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*in_ch, [0.5]*in_ch),
    ])

def load_indices(path: str) -> Optional[List[int]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def sample_per_class(imfolder: ImageFolder, n_per_class: int, seed: int) -> List[int]:
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

# ---------- Decode helper ----------
def _decode_flat(decoder: nn.Module, z_like: torch.Tensor, z_flat: torch.Tensor, SCALE: float) -> torch.Tensor:
    """
    decoder is AutoencoderKL; we only use decode.
    Expect z_like shape: (1, Cz, Hz, Wz); z_flat is flattened latent.
    Returns flattened output (M,) for use with autograd.grad VJP calls.
    """
    z = z_flat.view_as(z_like) / SCALE
    x = decoder.decode(z).sample  # [-1,1], differentiable
    return x.reshape(-1)          # (M,)

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

# ---------- Per-dimension contributions via Hutchinson estimator ----------
def compute_perdim_for_latents(
    vae: AutoencoderKL,
    Z: torch.Tensor,
    device: torch.device,
    SCALE: float,
    n_mc: int = 64,
    eps: float = 1e-12,
    desc: str = "per-dim contrib"
) -> np.ndarray:
    """
    For each z in Z (B,C,H,W), estimate per-dimension contributions:
        c_k(z) = 0.5 * log( (J^T J)_{kk} + eps )
    where J is the Jacobian of g(z) = decode(z / SCALE).sample wrt z (flattened).
    Hutchinson: diag(J^T J) = E_v[ (J^T v) ⊙ (J^T v) ], v ~ N(0, I).
    Returns array of shape (N, D) with D = C*H*W.
    """
    vae = vae.to(device).eval()
    N = Z.shape[0]
    C, H, W = Z.shape[1], Z.shape[2], Z.shape[3]
    D = C * H * W

    results = []
    for i in tqdm(range(N), desc=desc):
        # Keep single-sample latent in fp32 for stable AD
        z_like = Z[i:i+1].to(device).to(torch.float32).detach()
        z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)

        # Build output once to reuse graph across MC probes
        out_flat = _decode_flat(vae, z_like, z_flat, SCALE)  # (M,)

        diag_est = torch.zeros(D, device=device, dtype=out_flat.dtype)

        # Multiple random probes v ~ N(0, I_M); accumulate (J^T v)^2
        for j in range(int(n_mc)):
            v = torch.randn_like(out_flat)  # (M,)
            g = torch.autograd.grad(
                outputs=out_flat,
                inputs=z_flat,
                grad_outputs=v,
                retain_graph=(j < n_mc - 1),
                create_graph=False,
                only_inputs=True,
                allow_unused=False,
            )[0]  # (D,)
            diag_est += g.pow(2)

        diag_est /= float(max(1, int(n_mc)))
        per_dim_log = 0.5 * torch.log(diag_est + eps)  # (D,)
        results.append(per_dim_log.detach().cpu())

        # cleanup
        z_flat.requires_grad_(False)
        del out_flat, diag_est, per_dim_log
        torch.cuda.empty_cache()

    per_dim = torch.stack(results, dim=0)  # (N, D)
    return per_dim.numpy().astype(np.float32)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="ImageNet root with train/ and val/")
    ap.add_argument("--out_dir", required=True, help="Must match trainer out_dir (for indices/json)")
    ap.add_argument("--out_npz", required=True, help="Where to save the metrics NPZ")
    ap.add_argument("--sd_vae_repo", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument("--latent_ch", type=int, default=4)
    ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2025)
    # ap.add_argument("--samples_per_class", type=int, default=10, help="held-out val samples per class (<=0 uses ALL)")
    ap.add_argument("--pullback_npz", type=str, required=True)
    # Kept for compatibility (not used in per-dim mode)
    ap.add_argument("--amp", action="store_true")
    # New estimator controls:
    ap.add_argument("--n_mc", type=int, default=8, help="MC probes for Hutchinson diag(J^T J)")
    ap.add_argument("--eps", type=float, default=1e-12, help="stability epsilon inside log")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.amp else torch.float32
    SCALE = 0.18215  # must match training usage

    # --- Datasets (same transforms as trainer) ---
    tx = build_transforms(args.img_size, args.in_ch)
    ds_train = ImageFolder(os.path.join(args.data_root, "train"), transform=tx)
    ds_val   = ImageFolder(os.path.join(args.data_root, "val"),   transform=tx)

    # --- Index selection (example policy: per-class sampling if >0) ---
    # if args.samples_per_class and args.samples_per_class > 0:
    #     member_indices  = sample_per_class(ds_train, args.samples_per_class, args.seed)
    #     heldout_indices = sample_per_class(ds_val,   args.samples_per_class, args.seed)
    # else:
    #     member_indices  = list(range(len(ds_train.samples)))
    #     heldout_indices = list(range(len(ds_val.samples)))

    # --- Subsets + Loaders ---
    d = load_pullback_npz(args.pullback_npz)
    member_indices, heldout_indices = d['member_indices'], d['heldout_indices']
    train_subset = Subset(ds_train, member_indices)
    val_subset   = Subset(ds_val,   heldout_indices)
    train_loader = DataLoader(train_subset, batch_size=args.encode_batch,
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=args.encode_batch,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f"[DataLoader] train(member)={len(train_subset)} | val(heldout)={len(val_subset)}")

    # --- Load SD VAE exactly like in trainer ---
    vae = AutoencoderKL.from_pretrained(args.sd_vae_repo, torch_dtype=dtype).to(device)
    vae.eval().requires_grad_(False)  # weights frozen during metric

    # Quick latent shape assert
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device, dtype=dtype)
        posterior = vae.encode(dummy).latent_dist
        z = posterior.mean * SCALE
        C, H, W = z.shape[1], z.shape[2], z.shape[3]
        assert (C, H, W) == (args.latent_ch, args.img_size//8, args.img_size//8), \
            f"latent shape mismatch: {(C,H,W)} vs {(args.latent_ch, args.img_size//8, args.img_size//8)}"

    # AMP still used for encoding only (as before)
    autocast_ctx = torch.cuda.amp.autocast if (device.type == "cuda" and args.amp) else torch.cpu.amp.autocast

    # --- Encode μ(x)*SCALE for selected indices ---
    def encode_mu_scaled(loader: DataLoader) -> torch.Tensor:
        zs = []
        for imgs, _ in tqdm(loader, desc="encode μ(x)*SCALE"):
            imgs = imgs.to(device)
            with torch.no_grad(), autocast_ctx():
                mu = vae.encode(imgs).latent_dist.mean
                z  = (mu * SCALE).to(torch.float32)
            zs.append(z.cpu())
        return torch.cat(zs, dim=0)

    print("[Encode] Train μ(x)*SCALE…")
    Z_train = encode_mu_scaled(train_loader)
    print("[Encode] Val μ(x)*SCALE…")
    Z_val   = encode_mu_scaled(val_loader)

    print("Latent shapes:", Z_train.shape, Z_val.shape)

    # --- Compute per-dimension contributions via Hutchinson ---
    print(f"[Per-dim] train (N={Z_train.shape[0]}), n_mc={args.n_mc}")
    member_perdim = compute_perdim_for_latents(
        vae=vae, Z=Z_train, device=device, SCALE=SCALE,
        n_mc=args.n_mc, eps=args.eps, desc="train per-dim"
    )
    print(f"[Per-dim] val   (N={Z_val.shape[0]}), n_mc={args.n_mc}")
    heldout_perdim = compute_perdim_for_latents(
        vae=vae, Z=Z_val, device=device, SCALE=SCALE,
        n_mc=args.n_mc, eps=args.eps, desc="val per-dim"
    )

    # --- Save ---
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    meta: Dict = dict(
        data_root=os.path.abspath(args.data_root),
        out_dir=os.path.abspath(args.out_dir),
        sd_vae_repo=args.sd_vae_repo,
        img_size=args.img_size,
        latent_ch=args.latent_ch,
        n_mc=args.n_mc,
        eps=args.eps,
        amp=args.amp,
        seed=args.seed,
        note="Per-dim c_k = 0.5*log(diag(J^T J)_k + eps) at z=mu(x)*SCALE; J from decode(z/SCALE).sample.",
    )
    np.savez_compressed(
        args.out_npz,
        member_indices=np.asarray(member_indices, dtype=np.int64),
        member_per_dim_logcontrib=member_perdim.astype(np.float32),
        heldout_indices=np.asarray(heldout_indices, dtype=np.int64),
        heldout_per_dim_logcontrib=heldout_perdim.astype(np.float32),
        meta=meta,
    )
    print(f"[Done] saved -> {args.out_npz}")

if __name__ == "__main__":
    main()
