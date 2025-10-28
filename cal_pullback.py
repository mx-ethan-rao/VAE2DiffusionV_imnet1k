#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cal_pullback_metric.py

Compute per-sample pullback metric from the SD VAE decoder used in main.py:
  - z(x) = μ(x) * SCALE, where μ is VAE encoder mean, SCALE=0.18215
  - decoder mapping g(z) = decode(z / SCALE).sample  (pixel space in [-1, 1])
  - Jacobian J(z) = ∂g/∂z  (flattened output wrt flattened latent)
Metric: sum_{i=1..k} log σ_i(J)  (top-k log-volume proxy by default)

Outputs (NPZ):
  - member_indices      : indices into ImageFolder(train).samples (balanced subset from main.py)
  - member_metric       : log-volume metric per chosen train sample
  - heldout_indices     : indices into ImageFolder(val).samples (configurable sampling)
  - heldout_metric      : log-volume metric per chosen val sample
  - meta                : dict with config

Example:
  python cal_pullback_metric.py \
    --data_root /path/imagenet \
    --out_dir runs/ldm_imnet256_sdvae \
    --out_npz runs/ldm_imnet256_sdvae/pullback_k16_logvol.npz \
    --sd_vae_repo stabilityai/sd-vae-ft-mse \
    --img_size 256 --encode_batch 64 \
    --method jvp --k 16 --n_iter 8
"""

import os, json, argparse, random
from typing import List, Optional, Dict
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian, jvp
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Subset
# from torch.backends.cuda import sdp_kernel
# SDPA context (new API first, fallback to old if needed)
try:
    from torch.nn.attention import sdpa_kernel  # PyTorch ≥ 2.4
except Exception:
    from torch.backends.cuda import sdp_kernel  # deprecated fallback

# --- SDPA math-kernel compatibility wrapper (handles new & old signatures) ---
import contextlib, inspect

def sdpa_math_kernel():
    """
    Returns a context manager that forces math SDPA (disables flash/mem-efficient kernels).
    Works across PyTorch versions by introspecting the callable signature.
    Falls back to a no-op if SDPA ctx is unavailable (e.g., CPU).
    """
    # Try torch.nn.attention.sdpa_kernel (newer PyTorch)
    try:
        from torch.nn.attention import sdpa_kernel as _sk
        params = set(inspect.signature(_sk).parameters.keys())
        if {"use_flash", "use_mem_efficient", "use_math"} <= params:
            return _sk(use_flash=False, use_mem_efficient=False, use_math=True)
        if {"enable_flash", "enable_mem_efficient", "enable_math"} <= params:
            return _sk(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        # Last resort: try any single 'math' flag name that might exist
        if "use_math" in params:
            return _sk(use_math=True)
        if "enable_math" in params:
            return _sk(enable_math=True)
    except Exception:
        pass

    # Try torch.backends.cuda.sdp_kernel (older API)
    try:
        from torch.backends.cuda import sdp_kernel as _sk
        params = set(inspect.signature(_sk).parameters.keys())
        if {"use_flash", "use_mem_efficient", "use_math"} <= params:
            return _sk(use_flash=False, use_mem_efficient=False, use_math=True)
        if {"enable_flash", "enable_mem_efficient", "enable_math"} <= params:
            return _sk(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        if "use_math" in params:
            return _sk(use_math=True)
        if "enable_math" in params:
            return _sk(enable_math=True)
    except Exception:
        pass

    # No SDPA ctx available (CPU or very old build): no-op
    @contextlib.contextmanager
    def _null():
        yield
    return _null()





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

# ---------- Jacobian helpers (from your previous cal_volume, adapted) ----------
def _decode_flat(decoder: nn.Module, z_like: torch.Tensor, z_flat: torch.Tensor, SCALE: float) -> torch.Tensor:
    """
    decoder is AutoencoderKL; we only use decode.
    Expect z_like shape: (1, Cz, Hz, Wz); z_flat is flattened latent.
    """
    z = z_flat.view_as(z_like) / SCALE
    x = decoder.decode(z).sample  # [-1,1], differentiable
    return x.reshape(-1)          # (M,)

@torch.no_grad()
def topk_singular_values_power(J: torch.Tensor, k=8, n_iter=10) -> torch.Tensor:
    device = J.device
    d = J.shape[1]
    k = min(k, d)
    Q = torch.randn(d, k, device=device)
    for _ in range(n_iter):
        Z = J @ Q
        Q = J.T @ Z
        Q, _ = torch.linalg.qr(Q, mode='reduced')
    Z = J @ Q
    B = Q.T @ (J.T @ Z)
    evals = torch.linalg.eigvalsh(B).real.clamp_min(1e-12)
    evals, _ = torch.sort(evals, descending=True)
    return torch.sqrt(evals[:k])

def J_times_V_jvp(decoder_mean, z_flat, V):
    """Y = J(z) @ V via JVP; Shapes: z_flat(d,), V(d,r) -> Y(M,r)."""
    z0 = z_flat.detach().requires_grad_(True)
    r = V.shape[1]
    cols = []
    for j in range(r):
        _, Jv = jvp(lambda zz: decoder_mean(zz), (z0,), (V[:, j],))
        cols.append(Jv.detach())
    return torch.stack(cols, dim=1)  # (M, r)

def JT_times_Y_vjp(decoder_mean, z_flat, Y):
    """G = J(z)^T @ Y using VJP; Y(M,r) -> G(d,r)."""
    z0 = z_flat.detach().requires_grad_(True)
    mu = decoder_mean(z0)  # (M,)
    r = Y.shape[1]
    cols = []
    for j in range(r):
        y = Y[:, j]
        s = torch.dot(mu, y)  # scalar
        (grad_z,) = torch.autograd.grad(s, z0, retain_graph=(j < r - 1), allow_unused=False)
        cols.append(grad_z.detach())
    return torch.stack(cols, dim=1)  # (d, r)

# @torch.no_grad()
def topk_svals_power_matrixfree(decoder_mean, z_like: torch.Tensor, k=16, n_iter=8, use_jvp=True, eps_fd=1e-3) -> torch.Tensor:
    """
    Subspace iteration on G = J^T J using JVP/VJP or finite differences for J·v.
    """
    device = z_like.device
    z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
    d = z_flat.numel()
    k = min(k, d)

    def J_times_V_fd(z0_flat, V):
        h = eps_fd
        outs = []
        for j in range(V.shape[1]):
            v = V[:, j]
            y1 = decoder_mean(z0_flat + h*v)
            y2 = decoder_mean(z0_flat - h*v)
            outs.append(((y1 - y2) / (2*h)).detach())
        return torch.stack(outs, dim=1)

    Jdot = (lambda V: J_times_V_jvp(decoder_mean, z_flat, V)) if use_jvp else (lambda V: J_times_V_fd(z_flat, V))
    JTdot = lambda Y: JT_times_Y_vjp(decoder_mean, z_flat, Y)

    Q = torch.randn(d, k, device=device)
    Q, _ = torch.linalg.qr(Q, mode='reduced')
    for _ in range(n_iter):
        JQ = Jdot(Q)      # (M, k)
        Y  = JTdot(JQ)    # (d, k)
        Q, _ = torch.linalg.qr(Y, mode='reduced')

    JQ = Jdot(Q)
    B = JQ.T @ JQ
    evals = torch.linalg.eigvalsh(B).real.clamp_min(1e-12)
    evals, _ = torch.sort(evals, descending=True)
    return torch.sqrt(evals[:k])

def topk_svals_randomized(
    decoder_mean,
    z_like: torch.Tensor,
    k: int,
    oversample: int = 10,
    power: int = 1,
    use_jvp: bool = True,
    eps_fd: float = 1e-3,
) -> torch.Tensor:
    """
    Randomized SVD for J to get top-k singular values with ~2*(k+oversample) applications.
    Uses same JVP/VJP primitives as power method but with far fewer decoder calls.

    Steps:
      1) V0 = randn(d, l), l = min(k+oversample, d)
      2) Optional power passes on (J^T J): for t in 1..power:
             Y = J * V;  V = orth(J^T * Y)
      3) Y = J * V; Q = orth(Y)  (M x l)
      4) T = J^T * Q  (d x l)
      5) s = svdvals(T)[:k]
    """
    device = z_like.device
    z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
    d = z_flat.numel()
    l = int(min(k + oversample, d))
    if l <= 0:
        return torch.zeros(0, device=device)

    # define J· and J^T· (reuse your helpers)
    def J_times_V_fd(z0_flat, V):
        h = eps_fd
        outs = []
        for j in range(V.shape[1]):
            v = V[:, j]
            y1 = decoder_mean(z0_flat + h*v)
            y2 = decoder_mean(z0_flat - h*v)
            outs.append(((y1 - y2) / (2*h)).detach())
        return torch.stack(outs, dim=1)

    Jdot = (lambda V: J_times_V_jvp(decoder_mean, z_flat, V)) if use_jvp else (lambda V: J_times_V_fd(z_flat, V))
    JTdot = lambda Y: JT_times_Y_vjp(decoder_mean, z_flat, Y)

    # 1) Start with random subspace in latent space
    V = torch.randn(d, l, device=device)
    V, _ = torch.linalg.qr(V, mode='reduced')

    # 2) Optional power iterations on (J^T J) to sharpen spectrum
    for _ in range(max(0, int(power))):
        Y = Jdot(V)          # (M, l)
        G = JTdot(Y)         # (d, l)
        V, _ = torch.linalg.qr(G, mode='reduced')

    # 3) Form image-space basis and orthonormalize
    Y = Jdot(V)              # (M, l)
    Q, _ = torch.linalg.qr(Y, mode='reduced')  # (M, l)

    # 4) Small dxl matrix and 5) SVD there
    T = JTdot(Q)             # (d, l)
    svals = torch.linalg.svdvals(T).clamp_min(1e-12)  # (l,)
    svals, _ = torch.sort(svals, descending=True)
    return svals[:k]


# ---------- Metric ----------
def compute_logvol_for_latents(
    vae: AutoencoderKL,
    Z: torch.Tensor,
    device: torch.device,
    SCALE: float,
    k: int,
    n_iter: int,
    method: str,
    eps_fd: float,
    desc: str,
) -> np.ndarray:
    """
    For each z in Z (B,C,H,W), compute sum(log σ_i(J(z))) with J the Jacobian of
    x = decode(z / SCALE).sample wrt z.
    """
    vae = vae.to(device).eval()
    N = Z.shape[0]
    out = []

    for i in tqdm(range(N), desc=desc):
        z_like = Z[i:i+1].to(device)  # keep (1,C,H,W)

        # def decoder_mean(zz_flat):
        #     return _decode_flat(vae, z_like, zz_flat, SCALE)

        def decoder_mean(zz_flat):
            # Force fp32 + math SDPA
            with torch.cuda.amp.autocast(enabled=False):
                with sdpa_math_kernel():
                    return _decode_flat(vae, z_like.to(torch.float32), zz_flat.to(torch.float32), SCALE)




        if method == "explicit":
            # WARNING: very large M and d, only for tiny tests
            z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
            # J = jacobian(lambda zz: decoder_mean(zz), z_flat)  # (M,d)
            with torch.cuda.amp.autocast(enabled=False):
                with sdpa_math_kernel():
                    J = jacobian(lambda zz: decoder_mean(zz), z_flat)  # (M,d)


            # svals = torch.linalg.svdvals(J)[:k].clamp_min(1e-12)
            svals = topk_singular_values_power(J=J, k=k, n_iter=n_iter).clamp_min(1e-12)
        elif method == "jvp":
            svals = topk_svals_power_matrixfree(
                decoder_mean, z_like, k=k, n_iter=n_iter,
                use_jvp=(method == "jvp"), eps_fd=eps_fd
            ).clamp_min(1e-12)
        elif method == "fd":
            svals = topk_svals_randomized(
                decoder_mean, z_like,
                k=k,
                oversample=30,     # small oversampling for stability
                power=2,           # one power pass sharpens spectrum at low cost
                use_jvp=(method == "jvp"),
                eps_fd=eps_fd,
            ).clamp_min(1e-12)

        logvol = torch.log(svals).sum().item()
        out.append(float(logvol))
        torch.cuda.empty_cache()

    return np.asarray(out, dtype=np.float32)

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
    ap.add_argument("--samples_per_class", type=int, default=10, help="held-out val samples per class (<=0 uses ALL)")
    ap.add_argument("--method", choices=["jvp","fd","explicit"], default="jvp")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--eps_fd", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.amp else torch.float32
    SCALE = 0.18215  # must match training usage
    # ^ main.py encodes z = μ(x) * SCALE and decodes with vae.decode(z / SCALE).sample  :contentReference[oaicite:3]{index=3}

    # --- Datasets (same transforms as trainer) ---
    tx = build_transforms(args.img_size, args.in_ch)
    ds_train = ImageFolder(os.path.join(args.data_root, "train"), transform=tx)
    ds_val   = ImageFolder(os.path.join(args.data_root, "val"),   transform=tx)

    # --- Member indices: load balanced subset the trainer saved ---
    # idx_json = os.path.join(args.out_dir, "train_indices_balanced.json")
    # member_indices = load_indices(idx_json)
    # if member_indices is None:
    #     raise FileNotFoundError(
    #         f"Could not find {idx_json}. Run training first (which writes it), "
    #         "or create it with the same sampling logic."
    #     )  # :contentReference[oaicite:4]{index=4}

    # --- Held-out indices: choose per class from val (or all) ---
    if args.samples_per_class and args.samples_per_class > 0:
        member_indices = sample_per_class(ds_train, args.samples_per_class, args.seed)
        heldout_indices = sample_per_class(ds_val, args.samples_per_class, args.seed)
    else:
        member_indices = list(range(len(ds_train.samples)))
        heldout_indices = list(range(len(ds_val.samples)))

    # --- Subsets + Loaders ---
    train_subset = Subset(ds_train, member_indices)
    val_subset   = Subset(ds_val, heldout_indices)
    train_loader = DataLoader(train_subset, batch_size=args.encode_batch,
                            shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=args.encode_batch,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"[DataLoader] train(member)={len(train_subset)} | val(heldout)={len(val_subset)}")


    # --- Load SD VAE exactly like in trainer ---
    vae = AutoencoderKL.from_pretrained(args.sd_vae_repo, torch_dtype=dtype).to(device)
    vae.eval().requires_grad_(False)  # weights frozen during metric
    # quick latent shape assert (mirrors trainer)  :contentReference[oaicite:5]{index=5}
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device, dtype=dtype)
        posterior = vae.encode(dummy).latent_dist
        z = posterior.mean * SCALE
        C, H, W = z.shape[1], z.shape[2], z.shape[3]
        assert (C, H, W) == (args.latent_ch, args.img_size//8, args.img_size//8), \
            f"latent shape mismatch: {(C,H,W)} vs {(args.latent_ch, args.img_size//8, args.img_size//8)}"

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

    # --- Compute metric (sum log top-k svals of decoder Jacobian) ---
    print(f"[Metric] train (N={Z_train.shape[0]}), method={args.method}, k={args.k}, n_iter={args.n_iter}")
    metric_train = compute_logvol_for_latents(
        vae=vae, Z=Z_train, device=device, SCALE=SCALE,
        k=args.k, n_iter=args.n_iter, method=args.method, eps_fd=args.eps_fd,
        desc="train logvol"
    )
    print(f"[Metric] val   (N={Z_val.shape[0]}), method={args.method}, k={args.k}, n_iter={args.n_iter}")
    metric_val = compute_logvol_for_latents(
        vae=vae, Z=Z_val, device=device, SCALE=SCALE,
        k=args.k, n_iter=args.n_iter, method=args.method, eps_fd=args.eps_fd,
        desc="val logvol"
    )

    # --- Save ---
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    meta: Dict = dict(
        data_root=os.path.abspath(args.data_root),
        out_dir=os.path.abspath(args.out_dir),
        sd_vae_repo=args.sd_vae_repo,
        img_size=args.img_size,
        latent_ch=args.latent_ch,
        method=args.method, k=args.k, n_iter=args.n_iter, eps_fd=args.eps_fd,
        amp=args.amp, seed=args.seed,
        note="Jacobian wrt z of decode(z/SCALE).sample at z=mu(x)*SCALE; metric=sum(log top-k svals).",
    )
    np.savez_compressed(
        args.out_npz,
        member_indices=np.asarray(member_indices, dtype=np.int64),
        member_metric=metric_train.astype(np.float32),
        heldout_indices=np.asarray(heldout_indices, dtype=np.int64),
        heldout_metric=metric_val.astype(np.float32),
        meta=meta,
    )
    print(f"[Done] saved -> {args.out_npz}")

if __name__ == "__main__":
    main()
