"""
training/train.py
=================
Deep Learning Training Pipeline for Satellite Forest Canopy Segmentation.
For CLI usage, run either `python train.py` or `python -m training.train`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
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
ROOT_DIR = Path(__file__).resolve().parent.parent
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
            self.samples.append((f1, "t1", fm))
            self.samples.append((f2, "t2", fm))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, kind, mask_path = self.samples[idx]

        img = np.array(Image.open(img_path))[:, :, :3]
        deforest_mask = np.array(Image.open(mask_path))
        if deforest_mask.ndim == 3:
            deforest_mask = deforest_mask[:, :, 0]
        deforest_bin = (deforest_mask > 127).astype(np.float32)

        if kind == "t1":
            forest_mask = np.ones_like(deforest_bin, dtype=np.float32)
        else:
            forest_mask = (1.0 - deforest_bin).astype(np.float32)

        tw, th = self.img_size
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        forest_mask = cv2.resize(forest_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                forest_mask = np.fliplr(forest_mask).copy()

            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                forest_mask = np.flipud(forest_mask).copy()

            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k).copy()
                forest_mask = np.rot90(forest_mask, k=rot_k).copy()

            bright_scale = np.random.uniform(0.92, 1.08)
            img = np.clip(img.astype(np.float32) * bright_scale, 0, 255).astype(np.uint8)

        img_norm = img.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(forest_mask).unsqueeze(0).float()

        return tensor_img, tensor_mask


def train_forest_segmentation_model(
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 1e-3,
    img_size: int = 256,
    no_augment: bool = False,
    checkpoint_name: str = "unet_deforestation.pth",
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Forest Segmentation U-Net training on {device}...")

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation" if (DATASET_DIR / "validation").exists() else DATASET_DIR / "val"

    train_ds = SatelliteForestDataset(train_dir, img_size=(img_size, img_size), augment=(not no_augment))
    val_ds = SatelliteForestDataset(val_dir, img_size=(img_size, img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ForestUNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_checkpoint = CHECKPOINT_DIR / checkpoint_name

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(best_checkpoint))

    print(f"Training Complete! Best model saved to: {best_checkpoint}")
    return best_checkpoint


if __name__ == "__main__":
    train_forest_segmentation_model()
