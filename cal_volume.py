#!/usr/bin/env python3
"""
cal_logvolume.py

Compute per-sample decoder-Jacobian log-volume for a VAE trained with your
imagenet_ldm_ddp.py, and save results for both ImageNet train (members) and
val (held-out) splits.

Output (NPZ):
  member_indices    : np.ndarray[int]   -> indices into ImageFolder(train).samples
  member_logvols    : np.ndarray[float] -> log-volume per chosen train sample
  heldout_indices   : np.ndarray[int]   -> indices into ImageFolder(val).samples
  heldout_logvols   : np.ndarray[float] -> log-volume per chosen val sample
  meta              : dict              -> config used for this run

Methods:
  - explicit : build full Jacobian J with torch.autograd.functional.jacobian
               (--explicit_solver svdvals|power, and --use_all_svals to sum all svals)
  - jvp      : matrix-free subspace iteration using JVP (J·v) and VJP (Jᵀ·y)
  - fd       : same as jvp but J·v via finite differences (for debugging)

Example:
  python cal_logvolume.py \
    --data_root /path/imagenet \
    --vae_ckpt runs/ldm_imnet256/vae_last.pt \
    --out_npz logvol_imnet256.npz \
    --img_size 256 --per_class_train 3 --per_class_val 3 \
    --method explicit --explicit_solver svdvals --k 32
"""

import os, math, json, argparse, random
from collections import defaultdict
from typing import Tuple, Optional, List

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian, jvp
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ------------------------------
# Utils
# ------------------------------
def seed_all(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms(img_size: int, in_ch: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*in_ch, [0.5]*in_ch),
    ])

def sample_per_class_indices(ds: ImageFolder, n_per_class: Optional[int], seed: int) -> List[int]:
    """
    Return balanced indices (n_per_class per class). If n_per_class is None, returns all indices.
    """
    if n_per_class is None:
        return list(range(len(ds.samples)))
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for idx, (_, lbl) in enumerate(ds.samples):
        buckets[lbl].append(idx)
    chosen = []
    for lbl, bucket in buckets.items():
        if len(bucket) < n_per_class:
            raise RuntimeError(f"class {lbl} has only {len(bucket)} images (< {n_per_class})")
        rng.shuffle(bucket)
        chosen.extend(bucket[:n_per_class])
    rng.shuffle(chosen)
    return chosen

# ------------------------------
# VAE (exactly like your training script)
# ------------------------------
class GN(nn.GroupNorm):
    def __init__(self, ch: int, max_groups: int = 32):
        g = math.gcd(ch, max_groups) or 1
        super().__init__(g, ch)

class ResBlockSimple(nn.Module):
    def __init__(self, ch, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            GN(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.Dropout(drop), GN(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)

class VAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4, base=128, drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base, 3, stride=2, padding=1),
            ResBlockSimple(base, drop),
            nn.Conv2d(base, base*2, 3, stride=2, padding=1),
            ResBlockSimple(base*2, drop),
        )
        self.to_mu     = nn.Conv2d(base*2, latent_ch, 1)
        self.to_logvar = nn.Conv2d(base*2, latent_ch, 1)
        self.dec_in = nn.Conv2d(latent_ch, base*2, 1)
        self.dec = nn.Sequential(
            ResBlockSimple(base*2, drop),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1),
            ResBlockSimple(base, drop),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),
            ResBlockSimple(base, drop),
            GN(base), nn.SiLU(), nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        return self.dec(self.dec_in(z))

# ------------------------------
# Jacobian utilities
# ------------------------------
def topk_singular_values_power(J: torch.Tensor, k=8, n_iter=10) -> torch.Tensor:
    """
    Estimate top-k singular values of J (M x d) using power iterations on J^T J.
    Returns svals in descending order (k,).
    """
    device = J.device
    d = J.shape[1]
    k = min(k, d)
    Q = torch.randn(d, k, device=device)
    for _ in range(n_iter):
        Z = J @ Q        # (M x k)
        Q = J.T @ Z      # (d x k)
        Q, _ = torch.linalg.qr(Q, mode='reduced')
    Z = J @ Q
    B = Q.T @ (J.T @ Z)          # (k x k)
    evals = torch.linalg.eigvalsh(B).real.clamp_min(1e-12)
    evals, _ = torch.sort(evals, descending=True)
    return torch.sqrt(evals[:k])

def _decode_flat(decoder: nn.Module, z_like: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
    x = decoder.decode(z_flat.view_as(z_like))  # (1, Cx, Hx, Wx)
    return x.reshape(-1)                        # (M,)

def J_times_V_jvp(decoder_mean, z_flat, V):
    """Y = J(z) @ V via JVP (forward-mode). Shapes: z_flat(d,), V(d,r) -> Y(M,r)."""
    z0 = z_flat.detach().requires_grad_(True)
    r = V.shape[1]
    cols = []
    for j in range(r):
        _, Jv = jvp(lambda zz: decoder_mean(zz), (z0,), (V[:, j],))
        cols.append(Jv.detach())
    return torch.stack(cols, dim=1)  # (M, r)

def JT_times_Y_vjp(decoder_mean, z_flat, Y):
    """G = J(z)^T @ Y using VJP: grad(<mu(z), y>). Shapes: Y(M,r) -> G(d,r)."""
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

def topk_svals_power_matrixfree(
    decoder_mean, z_like: torch.Tensor, k=16, n_iter=8, use_jvp=True, eps_fd=1e-3
) -> torch.Tensor:
    """
    Subspace iteration on G=J^T J using only J· and J^T·.
    If use_jvp=False, J·v via finite differences around z (for debugging).
    """
    device = z_like.device
    z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
    d = z_flat.numel()
    k = min(k, d)

    def J_times_V_fd(z0_flat, V):
        # central finite differences: [f(z+h v) - f(z-h v)] / (2h)
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
        JQ = Jdot(Q)          # (M, k)
        Y  = JTdot(JQ)        # (d, k)
        Q, _ = torch.linalg.qr(Y, mode='reduced')

    JQ = Jdot(Q)
    B = JQ.T @ JQ
    evals = torch.linalg.eigvalsh(B).real.clamp_min(1e-12)
    evals, _ = torch.sort(evals, descending=True)
    return torch.sqrt(evals[:k])

@torch.no_grad()
def svals_from_explicit_J(J: torch.Tensor, k: int, solver: str = "svdvals", use_all_svals: bool = False) -> torch.Tensor:
    if use_all_svals:
        return torch.linalg.svdvals(J)
    if solver == "svdvals":
        return torch.linalg.svdvals(J)[:k]
    elif solver == "power":
        return topk_singular_values_power(J, k=k, n_iter=10)
    else:
        raise ValueError("explicit_solver must be 'svdvals' or 'power'")

def compute_log_volume_per_sample_decoder(
    decoder: nn.Module,
    z_tensor: torch.Tensor,
    device: torch.device,
    k: int = 16,
    n_iter: int = 8,
    desc: str = "compute log-vol",
    method: str = "explicit",           # 'explicit' | 'jvp' | 'fd'
    explicit_solver: str = "svdvals",   # only for method='explicit'
    use_all_svals: bool = False,        # only for method='explicit'
    eps_fd: float = 1e-3,
) -> np.ndarray:
    """
    Log-volume proxy: sum(log σ_i(J(z))) over i=1..K (or all svals if use_all_svals).
    Returns np.array (N,).
    """
    decoder = decoder.to(device).eval()
    N = z_tensor.shape[0]
    results: List[float] = []

    for i in tqdm(range(N), desc=desc):
        z_like = z_tensor[i:i+1].to(device)               # keep latent shape
        if method == "explicit":
            z_flat = z_like.reshape(-1).detach().clone().requires_grad_(True)
            J = jacobian(lambda zz: _decode_flat(decoder, z_like, zz), z_flat)  # (M, d)
            J = J.reshape(J.shape[0], -1).to(device)
            svals = svals_from_explicit_J(J, k, solver=explicit_solver, use_all_svals=use_all_svals).clamp_min(1e-12)
            log_vol = torch.log(svals).sum().item()
            results.append(float(log_vol))
            del J, svals
        else:
            # matrix-free path
            def decoder_mean(zz_flat):
                return _decode_flat(decoder, z_like, zz_flat)
            svals = topk_svals_power_matrixfree(
                decoder_mean, z_like, k=k, n_iter=n_iter,
                use_jvp=(method == "jvp"), eps_fd=eps_fd
            ).clamp_min(1e-12)
            log_vol = torch.log(svals).sum().item()
            results.append(float(log_vol))

        torch.cuda.empty_cache()

    return np.array(results, dtype=np.float32)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help="ImageNet root with train/ and val/")
    ap.add_argument('--vae_ckpt',  type=str, required=True, help="Path to vae_last.pt from training")
    ap.add_argument('--out_npz',   type=str, required=True, help="Where to save the npz")

    ap.add_argument('--img_size',  type=int, default=256)
    ap.add_argument('--in_ch',     type=int, default=3)
    ap.add_argument('--latent_ch', type=int, default=4)
    ap.add_argument('--ae_base',   type=int, default=128)

    ap.add_argument('--per_class_train', type=int, default=3, help="#images per class from train (None=all)",)
    ap.add_argument('--per_class_val',   type=int, default=3, help="#images per class from val (None=all)",)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--encode_batch', type=int, default=64, help="batch size for encoding μ(x)")

    # log-volume params
    ap.add_argument('--method', choices=['explicit', 'jvp', 'fd'], default='explicit')
    ap.add_argument('--explicit_solver', choices=['svdvals', 'power'], default='svdvals')
    ap.add_argument('--use_all_svals', action='store_true', help="sum ALL singular values (explicit only)")
    ap.add_argument('--k', type=int, default=16, help="top-K singular values to sum")
    ap.add_argument('--n_iter', type=int, default=8, help="power/subspace iteration steps")
    ap.add_argument('--eps_fd', type=float, default=1e-3, help="finite-diff step for method=fd")

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- datasets & subsets ---
    tx = build_transforms(args.img_size, args.in_ch)
    ds_train = ImageFolder(os.path.join(args.data_root, "train"), transform=tx)
    ds_val   = ImageFolder(os.path.join(args.data_root, "val"),   transform=tx)

    train_idx = sample_per_class_indices(ds_train, None if args.per_class_train < 0 else args.per_class_train, args.seed)
    val_idx   = sample_per_class_indices(ds_val,   None if args.per_class_val   < 0 else args.per_class_val,   args.seed)

    print(f"[Data] train chosen: {len(train_idx)} | val chosen: {len(val_idx)}")

    # --- model ---
    vae = VAE(in_ch=args.in_ch, latent_ch=args.latent_ch, base=args.ae_base, drop=0.1).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval().requires_grad_(False)

    # --- encode μ(x) for all chosen images (batched, fast) ---
    def encode_mu(ds: ImageFolder, idxs: List[int]) -> torch.Tensor:
        zs: List[torch.Tensor] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(idxs), args.encode_batch), desc="encode μ(x)"):
                batch_idx = idxs[i:i+args.encode_batch]
                imgs = torch.stack([ds[i][0] for i in batch_idx], dim=0).to(device)
                mu, _ = vae.encode(imgs)   # (B, C, H/4, W/4) for your architecture
                zs.append(mu.cpu())
        return torch.cat(zs, dim=0)

    print("[Encode] Train μ(x)…")
    Z_train = encode_mu(ds_train, train_idx)
    print("[Encode] Val μ(x)…")
    Z_val   = encode_mu(ds_val,   val_idx)

    # --- compute log-volumes (per-sample) ---
    print(f"[LogVol] Train ({len(train_idx)}) with method={args.method}")
    lv_train = compute_log_volume_per_sample_decoder(
        decoder=vae, z_tensor=Z_train, device=device,
        k=args.k, n_iter=args.n_iter, method=args.method,
        explicit_solver=args.explicit_solver, use_all_svals=args.use_all_svals, eps_fd=args.eps_fd,
        desc="train log-vol"
    )

    print(f"[LogVol] Val ({len(val_idx)}) with method={args.method}")
    lv_val = compute_log_volume_per_sample_decoder(
        decoder=vae, z_tensor=Z_val, device=device,
        k=args.k, n_iter=args.n_iter, method=args.method,
        explicit_solver=args.explicit_solver, use_all_svals=args.use_all_svals, eps_fd=args.eps_fd,
        desc="val log-vol"
    )

    # --- save ---
    meta = dict(
        img_size=args.img_size, in_ch=args.in_ch, latent_ch=args.latent_ch, ae_base=args.ae_base,
        method=args.method, explicit_solver=args.explicit_solver, use_all_svals=args.use_all_svals,
        k=args.k, n_iter=args.n_iter, eps_fd=args.eps_fd, seed=args.seed,
        vae_ckpt=os.path.abspath(args.vae_ckpt),
        data_root=os.path.abspath(args.data_root),
    )
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        member_indices=np.array(train_idx, dtype=np.int64),
        member_logvols=lv_train.astype(np.float32),
        heldout_indices=np.array(val_idx, dtype=np.int64),
        heldout_logvols=lv_val.astype(np.float32),
        meta=meta,
    )
    print(f"[Done] Saved to {args.out_npz}")

if __name__ == "__main__":
    main()
