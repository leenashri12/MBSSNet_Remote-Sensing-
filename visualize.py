"""
Visualization Script — Produces Fig. 3-style colored segmentation maps.
========================================================================
Generates side-by-side comparisons:
   [SAR] [Optical(RGB)] [Ground Truth] [Prediction]

Usage:
    python visualize.py                              # uses best_model.pth
    python visualize.py --checkpoint best_model.pth  # specify checkpoint
    python visualize.py --num_samples 10             # number of images to generate
    python visualize.py --split test                 # which split to use
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MBSSNet
from dataset import MBSSNetDataset

# ── Color palette matching the paper's Fig. 3 legend ──────────
# Class index → (B, G, R) for OpenCV
CLASS_COLORS_BGR = {
    0: (193, 182, 255),   # Farmland  — Pink
    1: (0,   0,   255),   # City      — Red
    2: (0,   255, 255),   # Village   — Yellow
    3: (180, 0,   0),     # Water     — Dark Blue
    4: (0,   180, 0),     # Forest    — Green
    5: (255, 200, 0),    # Road      — Light Blue / Cyan
    6: (255, 0,   180),   # Others    — Purple
    7: (128, 128, 128),   # Background — Gray
}

CLASS_NAMES = [
    "Farmland", "City", "Village", "Water",
    "Forest", "Road", "Others", "Background"
]


def label_to_color(label_map):
    """
    Convert a (H, W) integer label map to a (H, W, 3) BGR color image.
    """
    H, W = label_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_BGR.items():
        mask = label_map == cls_id
        color_img[mask] = color
    return color_img


def draw_legend(height=40, width=900):
    """Draw the color legend bar matching the paper."""
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255
    n_classes = len(CLASS_NAMES)
    box_w = width // n_classes
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_BGR.values())):
        x1 = i * box_w
        x2 = x1 + box_w
        # Color box
        cv2.rectangle(legend, (x1 + 2, 4), (x1 + 22, 24), color, -1)
        cv2.rectangle(legend, (x1 + 2, 4), (x1 + 22, 24), (0, 0, 0), 1)
        # Label text
        cv2.putText(legend, name, (x1 + 26, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return legend


def visualize_sample(model, dataset, idx, device, save_dir):
    """Run inference on one sample and save the visualization."""
    opt, sar, lbl = dataset[idx]

    # Move to device and add batch dim
    opt_t = opt.unsqueeze(0).to(device)
    sar_t = sar.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(opt_t, sar_t)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    lbl_np = lbl.numpy()  # (H, W)

    # ── SAR visualization (1ch → grayscale → BGR) ──
    sar_np = (sar.squeeze(0).numpy() * 255).astype(np.uint8)
    sar_bgr = cv2.cvtColor(sar_np, cv2.COLOR_GRAY2BGR)

    # ── Optical visualization (take RGB channels, ignore NIR) ──
    opt_np = opt.numpy()  # (4, H, W)
    # Channels: 0=R, 1=G, 2=B, 3=NIR → use [2,1,0] for BGR
    opt_rgb = np.stack([opt_np[2], opt_np[1], opt_np[0]], axis=-1)  # (H,W,3)
    opt_bgr = (opt_rgb * 255).astype(np.uint8)

    # ── Color maps ──
    gt_color = label_to_color(lbl_np)
    pred_color = label_to_color(pred)

    # ── Add labels on top ──
    panels = []
    titles = ["(a) SAR", "(b) Optical", "(c) Label", "(d) Ours"]
    images = [sar_bgr, opt_bgr, gt_color, pred_color]

    for title, img in zip(titles, images):
        # Add title bar
        title_bar = np.ones((30, img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(title_bar, title, (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        panel = np.vstack([title_bar, img])
        panels.append(panel)

    # ── Combine side-by-side ──
    combined = np.hstack(panels)

    # ── Add legend at bottom ──
    legend = draw_legend(height=40, width=combined.shape[1])
    final = np.vstack([combined, legend])

    # ── Save ──
    fname = f"visualization_{idx:04d}.png"
    save_path = os.path.join(save_dir, fname)
    cv2.imwrite(save_path, final)
    print(f"  Saved: {save_path}")
    return pred, lbl_np


def compute_metrics(all_preds, all_labels, num_classes=8):
    """Compute OA and per-class IoU."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        for c_true in range(num_classes):
            for c_pred in range(num_classes):
                confusion[c_true, c_pred] += np.sum(
                    (label == c_true) & (pred == c_pred)
                )

    # Overall Accuracy
    oa = np.diag(confusion).sum() / confusion.sum()

    # Per-class IoU
    ious = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)

    miou = np.mean(ious)
    return oa, miou, ious


def main():
    parser = argparse.ArgumentParser(description="MBSSNet Visualization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, default="processed",
                        help="Path to processed data directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str, default="visualizations",
                        help="Directory to save output images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = MBSSNet(opt_channels=4, sar_channels=1, num_classes=8,
                    pretrained=False).to(device)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found at '{args.checkpoint}'")
        print("Running with RANDOM weights (predictions will be meaningless)")
        print("Train the model first, then re-run this script.\n")

    model.eval()

    # Dataset
    data_path = os.path.join(args.data_root, args.split)
    dataset = MBSSNetDataset(data_path, augment=False)
    print(f"Dataset: {len(dataset)} samples from '{args.split}' split")

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Select samples
    np.random.seed(args.seed)
    indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)),
                               replace=False)
    indices.sort()

    print(f"\nGenerating {len(indices)} visualizations...\n")

    all_preds = []
    all_labels = []

    for idx in indices:
        pred, lbl = visualize_sample(model, dataset, idx, device, args.save_dir)
        all_preds.append(pred)
        all_labels.append(lbl)

    # Compute metrics on visualized samples
    oa, miou, ious = compute_metrics(all_preds, all_labels)
    print(f"\n{'='*50}")
    print(f"Metrics on {len(indices)} visualized samples:")
    print(f"  Overall Accuracy (OA): {oa*100:.2f}%")
    print(f"  Mean IoU (mIoU):       {miou*100:.2f}%")
    print(f"\n  Per-class IoU:")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, ious)):
        print(f"    {i}: {name:12s} -> {iou*100:.2f}%")
    print(f"{'='*50}")
    print(f"\nAll visualizations saved to: {args.save_dir}/")
    print("You can use these images directly in your evaluation presentation!")


if __name__ == "__main__":
    main()
