"""
MBSSNet Data Preprocessing Pipeline
====================================
Crops large-scale TIFF images into 256x256 non-overlapping patches,
remaps label values to contiguous indices, splits into train/val/test (8:1:1),
and saves as uint8 (normalization deferred to DataLoader).

Usage:
    python preprocess_data.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import time


def main():
    # ──────────────────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────────────────
    ROOT = Path(r"c:\Users\LENOVO\Downloads\btp")
    OPTICAL_DIR = ROOT / "optical_data"
    SAR_DIR = ROOT / "sar_data"
    LABEL_DIR = ROOT / "label_data"
    OUTPUT_DIR = ROOT / "processed"
    PATCH_SIZE = 256
    SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    # TEST_RATIO = 0.1 (remainder)

    # ──────────────────────────────────────────────────────────
    # STEP 1: Filename Synchronization
    # ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  STEP 1: Filename Synchronization")
    print("=" * 64)

    opt_files = {f for f in os.listdir(OPTICAL_DIR) if f.lower().endswith('.tif')}
    sar_files = {f for f in os.listdir(SAR_DIR) if f.lower().endswith('.tif')}
    lbl_files = {f for f in os.listdir(LABEL_DIR) if f.lower().endswith('.tif')}

    common = sorted(opt_files & sar_files & lbl_files)

    print(f"  Optical folder : {len(opt_files)} files")
    print(f"  SAR folder     : {len(sar_files)} files")
    print(f"  Label folder   : {len(lbl_files)} files")
    print(f"  Common (paired): {len(common)} files")

    only_opt = opt_files - sar_files - lbl_files
    only_sar = sar_files - opt_files - lbl_files
    only_lbl = lbl_files - opt_files - sar_files
    if only_opt:
        print(f"  WARNING — only in optical: {only_opt}")
    if only_sar:
        print(f"  WARNING — only in SAR: {only_sar}")
    if only_lbl:
        print(f"  WARNING — only in label: {only_lbl}")

    assert len(common) > 0, "No common filenames found across the three folders!"
    print(f"  -> {len(common)} images perfectly paired across all three folders.\n")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Create Output Directories
    # ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  STEP 2: Creating Output Directories")
    print("=" * 64)

    splits = ["train", "val", "test"]
    modalities = ["optical", "sar", "label"]

    for split in splits:
        for mod in modalities:
            d = OUTPUT_DIR / split / mod
            d.mkdir(parents=True, exist_ok=True)
    print(f"  Output root: {OUTPUT_DIR}")
    print(f"  Subdirectories created for {splits} x {modalities}\n")

    # ──────────────────────────────────────────────────────────
    # STEP 3: Compute Patch Grid
    # ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  STEP 3: Computing Patch Grid")
    print("=" * 64)

    sample = cv2.imread(str(OPTICAL_DIR / common[0]), cv2.IMREAD_UNCHANGED)
    H, W = sample.shape[:2]
    n_rows = H // PATCH_SIZE
    n_cols = W // PATCH_SIZE
    patches_per_img = n_rows * n_cols
    total_patches = patches_per_img * len(common)

    print(f"  Image size     : {H} x {W}")
    print(f"  Patch size     : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"  Grid           : {n_rows} rows x {n_cols} cols = {patches_per_img} patches/image")
    print(f"  Discarded edge : {H % PATCH_SIZE}px bottom, {W % PATCH_SIZE}px right")
    print(f"  Total patches  : {total_patches}\n")

    # ──────────────────────────────────────────────────────────
    # STEP 4: Random Split (8:1:1)
    # ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  STEP 4: Random Split (8:1:1)")
    print("=" * 64)

    random.seed(SEED)
    np.random.seed(SEED)
    perm = np.random.permutation(total_patches)

    n_train = int(total_patches * TRAIN_RATIO)
    n_val = int(total_patches * VAL_RATIO)
    n_test = total_patches - n_train - n_val

    # Build a mapping: global_patch_index -> split name
    split_map = np.empty(total_patches, dtype='U5')
    split_map[perm[:n_train]] = "train"
    split_map[perm[n_train:n_train + n_val]] = "val"
    split_map[perm[n_train + n_val:]] = "test"

    print(f"  Train : {n_train}")
    print(f"  Val   : {n_val}")
    print(f"  Test  : {n_test}")
    print(f"  Total : {n_train + n_val + n_test}\n")

    # ──────────────────────────────────────────────────────────
    # STEP 5: Crop & Save Patches
    # ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  STEP 5: Cropping & Saving Patches (uint8)")
    print("=" * 64)

    counts = {s: {m: 0 for m in modalities} for s in splits}
    t0 = time.time()

    for img_idx, fname in enumerate(common):
        stem = Path(fname).stem
        elapsed = time.time() - t0
        print(f"  [{img_idx + 1:3d}/{len(common)}] {fname}  "
              f"({elapsed:.0f}s elapsed)", flush=True)

        # Read images
        opt_img = cv2.imread(str(OPTICAL_DIR / fname), cv2.IMREAD_UNCHANGED)  # (H, W, 4)
        sar_img = cv2.imread(str(SAR_DIR / fname), cv2.IMREAD_UNCHANGED)      # (H, W)
        lbl_img = cv2.imread(str(LABEL_DIR / fname), cv2.IMREAD_UNCHANGED)    # (H, W)

        # Remap labels:  0,10,20,...,70  ->  0,1,2,...,7
        lbl_img = (lbl_img // 10).astype(np.uint8)

        for r in range(n_rows):
            for c in range(n_cols):
                g_idx = img_idx * patches_per_img + r * n_cols + c
                split = split_map[g_idx]

                y0 = r * PATCH_SIZE
                x0 = c * PATCH_SIZE
                y1 = y0 + PATCH_SIZE
                x1 = x0 + PATCH_SIZE

                patch_name = f"{stem}_r{r:02d}_c{c:02d}.tif"

                cv2.imwrite(str(OUTPUT_DIR / split / "optical" / patch_name),
                            opt_img[y0:y1, x0:x1])
                cv2.imwrite(str(OUTPUT_DIR / split / "sar" / patch_name),
                            sar_img[y0:y1, x0:x1])
                cv2.imwrite(str(OUTPUT_DIR / split / "label" / patch_name),
                            lbl_img[y0:y1, x0:x1])

                counts[split]["optical"] += 1
                counts[split]["sar"] += 1
                counts[split]["label"] += 1

    elapsed_total = time.time() - t0
    print(f"\n  Cropping complete in {elapsed_total:.1f}s\n")

    # ──────────────────────────────────────────────────────────
    # STEP 6: Validation Report
    # ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  STEP 6: Validation Report")
    print("=" * 64)

    all_aligned = True
    for split in splits:
        oc = counts[split]["optical"]
        sc = counts[split]["sar"]
        lc = counts[split]["label"]
        ok = (oc == sc == lc)
        tag = "ALIGNED" if ok else "MISALIGNED"
        sym = "+" if ok else "!"
        print(f"  [{sym}] {split:5s}:  optical={oc:6d}  sar={sc:6d}  label={lc:6d}  -> {tag}")
        if not ok:
            all_aligned = False

    grand = sum(counts[s]["optical"] for s in splits)
    print(f"\n  Grand total : {grand}  (expected {total_patches})")
    print(f"  CMFM 1:1:1 alignment: {'MAINTAINED' if all_aligned else 'BROKEN'}")

    # Spot-check a random training patch
    print("\n  Spot-check (first train patch):")
    train_opt_dir = OUTPUT_DIR / "train" / "optical"
    sample_name = sorted(os.listdir(train_opt_dir))[0]
    so = cv2.imread(str(OUTPUT_DIR / "train" / "optical" / sample_name), cv2.IMREAD_UNCHANGED)
    ss = cv2.imread(str(OUTPUT_DIR / "train" / "sar" / sample_name), cv2.IMREAD_UNCHANGED)
    sl = cv2.imread(str(OUTPUT_DIR / "train" / "label" / sample_name), cv2.IMREAD_UNCHANGED)
    print(f"    File   : {sample_name}")
    print(f"    Optical: shape={so.shape}, dtype={so.dtype}, range=[{so.min()}, {so.max()}]")
    print(f"    SAR    : shape={ss.shape}, dtype={ss.dtype}, range=[{ss.min()}, {ss.max()}]")
    print(f"    Label  : shape={sl.shape}, dtype={sl.dtype}, unique={np.unique(sl)}")

    print("\n" + "=" * 64)
    print("  PREPROCESSING COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
