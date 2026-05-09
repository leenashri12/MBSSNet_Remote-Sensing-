"""
PyTorch Dataset for MBSSNet
============================
Loads preprocessed 256×256 patches (uint8) from the processed/ directory.
Normalizes optical and SAR to [0, 1] float32 at load time.
Labels are kept as int64 class indices (0–7).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MBSSNetDataset(Dataset):
    """
    Parameters
    ----------
    root       : path to a split directory, e.g. "processed/train"
    augment    : apply data augmentation (random flip + rotation)
    """

    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment

        opt_dir = os.path.join(root, "optical")
        self.filenames = sorted([
            f for f in os.listdir(opt_dir) if f.lower().endswith(".tif")
        ])

        self.opt_dir = opt_dir
        self.sar_dir = os.path.join(root, "sar")
        self.lbl_dir = os.path.join(root, "label")

        # Sanity check: ensure all three modalities have the same files
        sar_set = set(os.listdir(self.sar_dir))
        lbl_set = set(os.listdir(self.lbl_dir))
        for fn in self.filenames:
            assert fn in sar_set, f"Missing SAR patch: {fn}"
            assert fn in lbl_set, f"Missing label patch: {fn}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # ── Load images ──
        # Optical: (256, 256, 4) uint8 → float32 [0,1]
        opt = cv2.imread(os.path.join(self.opt_dir, fname),
                         cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        # SAR: (256, 256) uint8 → float32 [0,1]
        sar = cv2.imread(os.path.join(self.sar_dir, fname),
                         cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        # Label: (256, 256) uint8 → int64
        lbl = cv2.imread(os.path.join(self.lbl_dir, fname),
                         cv2.IMREAD_UNCHANGED).astype(np.int64)

        # ── Data augmentation ──
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                opt = np.flip(opt, axis=1).copy()
                sar = np.flip(sar, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                opt = np.flip(opt, axis=0).copy()
                sar = np.flip(sar, axis=0).copy()
                lbl = np.flip(lbl, axis=0).copy()
            # Random 90° rotation
            k = np.random.randint(0, 4)
            if k > 0:
                opt = np.rot90(opt, k, axes=(0, 1)).copy()
                sar = np.rot90(sar, k, axes=(0, 1)).copy()
                lbl = np.rot90(lbl, k, axes=(0, 1)).copy()

        # ── Convert to tensors ──
        # Optical: (H, W, 4) → (4, H, W)
        opt = torch.from_numpy(opt).permute(2, 0, 1).contiguous()

        # SAR: (H, W) → (1, H, W)
        sar = torch.from_numpy(sar).unsqueeze(0).contiguous()

        # Label: (H, W) → (H, W)  (no channel dim for CE loss)
        lbl = torch.from_numpy(lbl).contiguous()

        return opt, sar, lbl
