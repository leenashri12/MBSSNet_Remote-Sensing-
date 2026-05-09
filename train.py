"""
MBSSNet Training Script
========================
- Adam optimizer, LR = 1e-4, weight decay = 1e-4
- Cross-Entropy loss (pixel-wise, multi-class)
- 30 epochs
- Evaluates mIoU and OA on validation set each epoch

Usage:
    python train.py
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MBSSNet
from dataset import MBSSNetDataset


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(preds, labels, num_classes):
    """
    Compute Overall Accuracy (OA) and mean IoU (mIoU).
    preds, labels : numpy arrays of shape (N,) with class indices.
    """
    correct = (preds == labels).sum()
    total = labels.size
    oa = correct / total

    ious = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    miou = np.mean(ious) if ious else 0.0
    return oa, miou


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for opt_img, sar_img, label in loader:
        opt_img = opt_img.to(device)
        sar_img = sar_img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(opt_img, sar_img)        # (B, C, H, W)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    for opt_img, sar_img, label in loader:
        opt_img = opt_img.to(device)
        sar_img = sar_img.to(device)
        label = label.to(device)

        logits = model(opt_img, sar_img)
        loss = criterion(logits, label)
        running_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy().flatten()
        all_preds.append(preds)
        all_labels.append(label.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    oa, miou = compute_metrics(all_preds, all_labels, num_classes)
    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss, oa, miou


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train MBSSNet")
    parser.add_argument("--data_root", type=str, default="processed",
                        help="Path to processed/ directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--opt_channels", type=int, default=4)
    parser.add_argument("--sar_channels", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Datasets & DataLoaders ───────────────────────────────
    train_ds = MBSSNetDataset(os.path.join(args.data_root, "train"), augment=True)
    val_ds   = MBSSNetDataset(os.path.join(args.data_root, "val"),   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val   samples: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────
    model = MBSSNet(
        opt_channels=args.opt_channels,
        sar_channels=args.sar_channels,
        num_classes=args.num_classes,
        pretrained=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters  : {total_params:,}  (trainable: {trainable:,})")

    # ── Loss, Optimizer ──────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ── Checkpoint directory ─────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0.0
    start_epoch = 1

    # ── Resume from checkpoint if provided ───────────────────
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed at epoch {start_epoch}, best mIoU so far: {best_miou:.4f}")

    # ── Training loop ────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'OA':>8}  {'mIoU':>8}  {'Time':>7}  {'Best':>5}")
    print("=" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss, oa, miou = validate(model, val_loader, criterion,
                                      device, args.num_classes)

        elapsed = time.time() - t0
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_model.pth"))

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
        }, os.path.join(args.save_dir, "latest_checkpoint.pth"))

        star = " *" if is_best else ""
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  "
              f"{oa:8.4f}  {miou:8.4f}  {elapsed:6.1f}s{star}")

    print("=" * 72)
    print(f"Training complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
