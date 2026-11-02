"""
train.py
========
Deep Learning Training Pipeline for Satellite Forest Canopy Segmentation.

Trains ForestUNet with Combined BCE + Soft Dice Loss and satellite-safe data
augmentation (orthogonal rotations, horizontal/vertical flips, subtle radiometric
scaling). Evaluates validation loss and IoU after each epoch, saving the optimal
checkpoint to models/checkpoints/unet_deforestation.pth.

Usage:
    python train.py --epochs 6 --batch-size 8 --lr 0.001
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from models.unet import ForestUNet, CombinedBCEDiceLoss
from utils.metrics import compute_metrics

DATASET_DIR = ROOT_DIR / "dataset"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
WEIGHTS_DIR = ROOT_DIR / "weights"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


class SatelliteForestDataset(Dataset):
    """
    Dataset for single-image forest canopy segmentation with realistic satellite augmentations.
    
    Pairs:
    1. Time 1 (T1) baseline image -> Intact forest canopy ground truth (mask = 1.0)
    2. Time 2 (T2) post-event image -> Intact canopy minus detected deforestation
    """
    def __init__(
        self,
        split_dir: str | Path,
        img_size: tuple[int, int] = (256, 256),
        augment: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.img_size = img_size
        self.augment = augment

        t1_files = sorted(glob.glob(str(self.split_dir / "t1" / "*.png")))
        t2_files = sorted(glob.glob(str(self.split_dir / "t2" / "*.png")))
        mask_files = sorted(glob.glob(str(self.split_dir / "masks" / "*.png")))

        self.samples: list[tuple[str, str, str]] = []
        for f1, f2, fm in zip(t1_files, t2_files, mask_files):
            # T1 sample (baseline forest)
            self.samples.append((f1, "t1", fm))
            # T2 sample (post-event with clearing)
            self.samples.append((f2, "t2", fm))

        # Ingest authentic online satellite dataset (Forest-Change) if downloaded
        split_key = self.split_dir.name
        if split_key == "validation":
            split_key = "val"
        online_split = DATASET_DIR / "online_forest_change" / "Forest-Change-dataset" / "images" / split_key
        if online_split.exists():
            online_a = sorted(glob.glob(str(online_split / "A" / "*.png")))
            online_b = sorted(glob.glob(str(online_split / "B" / "*.png")))
            online_m = sorted(glob.glob(str(online_split / "label" / "*.png")))
            for f1, f2, fm in zip(online_a, online_b, online_m):
                self.samples.append((f1, "t1", fm))
                self.samples.append((f2, "t2", fm))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, kind, mask_path = self.samples[idx]

        # Load optical RGB
        img = np.array(Image.open(img_path))[:, :, :3]
        deforest_mask = np.array(Image.open(mask_path))
        if deforest_mask.ndim == 3:
            deforest_mask = deforest_mask[:, :, 0]
        deforest_bin = (deforest_mask > 127).astype(np.float32)

        # Forest canopy ground-truth definition:
        # In T1 (pre-clearing), entire baseline forest is present (1.0)
        # In T2 (post-clearing), deforested areas are 0.0, intact forest remains 1.0
        if kind == "t1":
            forest_mask = np.ones_like(deforest_bin, dtype=np.float32)
        else:
            forest_mask = (1.0 - deforest_bin).astype(np.float32)

        # Geometric resizing
        tw, th = self.img_size
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        forest_mask = cv2.resize(forest_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Satellite-safe data augmentation (strictly preserving nadir perspective geometry)
        if self.augment:
            # 1. Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                forest_mask = np.fliplr(forest_mask).copy()

            # 2. Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                forest_mask = np.flipud(forest_mask).copy()

            # 3. Orthogonal 90° rotations (satellite nadir images have no true 'up')
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k).copy()
                forest_mask = np.rot90(forest_mask, k=rot_k).copy()

            # 4. Subtle radiometric solar illumination variation (±8%)
            bright_scale = np.random.uniform(0.92, 1.08)
            img = np.clip(img.astype(np.float32) * bright_scale, 0, 255).astype(np.uint8)

        # Radiometric normalization to [0.0, 1.0]
        img_norm = img.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(forest_mask).unsqueeze(0).float()

        return tensor_img, tensor_mask


def train_model(
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 1e-3,
    img_size: int = 256,
    no_augment: bool = False,
    checkpoint_name: str = "unet_deforestation.pth",
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Optimize CPU threads for multi-core parallelism
        torch.set_num_threads(min(12, os.cpu_count() or 8))
    print("=" * 65, flush=True)
    print(f"[TRAIN] ForestUNet Segmentation Model on Device: {device}", flush=True)
    print(f"        Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}", flush=True)
    print(f"        Augmentation: {not no_augment} | Target Image Size: {img_size}x{img_size}", flush=True)
    print("=" * 65, flush=True)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation" if (DATASET_DIR / "validation").exists() else DATASET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Dataset directories not found in {DATASET_DIR}. Run generate_dataset.py first.", flush=True)
        sys.exit(1)

    train_ds = SatelliteForestDataset(train_dir, img_size=(img_size, img_size), augment=(not no_augment))
    val_ds = SatelliteForestDataset(val_dir, img_size=(img_size, img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: {len(train_ds)} train samples | {len(val_ds)} validation samples", flush=True)

    model = ForestUNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for b_idx, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if b_idx % 5 == 0 or b_idx == len(train_loader):
                print(f"   -> Epoch {epoch:02d}/{epochs:02d} | Batch {b_idx:02d}/{len(train_loader):02d} | Step Loss: {loss.item():.4f}", flush=True)

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()

                pred_bin = (preds.cpu().numpy() >= 0.5).astype(np.uint8)
                gt_bin = (masks.cpu().numpy() >= 0.5).astype(np.uint8)
                for pb, gb in zip(pred_bin, gt_bin):
                    val_ious.append(compute_metrics(pb, gb)["iou"])

        val_loss /= len(val_loader)
        mean_val_iou = float(np.mean(val_ious))
        scheduler.step(val_loss)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(mean_val_iou, 4),
        }
        history.append(epoch_stats)

        is_best = val_loss < best_val_loss
        best_marker = "[BEST SAVED]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val IoU: {mean_val_iou:.4f} {best_marker}",
            flush=True,
        )

        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(best_checkpoint_path))
            # Also keep weights/unet_best.pt synchronized
            torch.save(model.state_dict(), str(WEIGHTS_DIR / "unet_best.pt"))

    # Save training history
    history_file = WEIGHTS_DIR / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[OK] Training completed successfully!")
    print(f"     Optimal Checkpoint: {best_checkpoint_path}")
    print(f"     Training History:   {history_file}")
    print("=" * 65)
    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train U-Net Forest Canopy Segmentation Model")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs (default: 6)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--img-size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--checkpoint-name", type=str, default="unet_deforestation.pth", help="Checkpoint filename")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        no_augment=args.no_augment,
        checkpoint_name=args.checkpoint_name,
    )


if __name__ == "__main__":
    main()
