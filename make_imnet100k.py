#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export balanced ImageNet subset based on saved indices JSON.
Reads from:  /path/to/imagenet/train/
Writes to:   /path/to/output/train/
"""
import os, json, shutil
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def export_subset(imnet_root: str, index_json: str, out_root: str):
    # Original ImageNet training folder
    train_dir = os.path.join(imnet_root, "train")
    ds = ImageFolder(train_dir)  # just for file paths & labels

    # Load indices
    with open(index_json, "r") as f:
        indices = json.load(f)
    print(f"[Loaded] {len(indices)} images from {index_json}")

    # Prepare output structure
    out_train = os.path.join(out_root, "train")
    os.makedirs(out_train, exist_ok=True)

    # Copy subset
    for idx in tqdm(indices, desc="Copying images"):
        src_path, label = ds.samples[idx]
        class_name = ds.classes[label]
        dst_dir = os.path.join(out_train, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)

    print(f"[Done] Exported {len(indices)} images to {out_train}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="Path to original ImageNet root (contains 'train' folder).")
    parser.add_argument("--index_json", type=str, required=True,
                        help="Path to train_indices_balanced.json.")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Path to output folder where subset will be saved.")
    args = parser.parse_args()

    export_subset(args.imagenet_root, args.index_json, args.out_root)
