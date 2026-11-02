"""
train_rf.py
===========
Trains the Random Forest multispectral deforestation classifier on sampled pixels
from the training dataset.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from models.random_forest import RandomForestDeforestationDetector

def train_random_forest():
    print("=" * 65)
    print("🌲 Training Random Forest Classifier on Multispectral Features...")
    print("=" * 65)

    rf = RandomForestDeforestationDetector(n_estimators=100, max_depth=16, random_state=42)
    t1_files = sorted(glob.glob(str(ROOT_DIR / "dataset" / "train" / "t1" / "*.png")))
    t2_files = sorted(glob.glob(str(ROOT_DIR / "dataset" / "train" / "t2" / "*.png")))
    mask_files = sorted(glob.glob(str(ROOT_DIR / "dataset" / "train" / "masks" / "*.png")))

    online_train = ROOT_DIR / "dataset" / "online_forest_change" / "Forest-Change-dataset" / "images" / "train"
    if online_train.exists():
        t1_files.extend(sorted(glob.glob(str(online_train / "A" / "*.png")))[:100])
        t2_files.extend(sorted(glob.glob(str(online_train / "B" / "*.png")))[:100])
        mask_files.extend(sorted(glob.glob(str(online_train / "label" / "*.png")))[:100])

    if not t1_files:
        print("❌ No training images found in dataset/train/t1/")
        return

    X_list, y_list = [], []
    np.random.seed(42)

    for f1, f2, fm in zip(t1_files, t2_files, mask_files):
        t1 = np.array(Image.open(f1))
        t2 = np.array(Image.open(f2))
        mask = (np.array(Image.open(fm)) > 127).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        X_full = rf.extract_features(t1, t2)
        y_full = mask.flatten()

        pos_idx = np.where(y_full == 1)[0]
        neg_idx = np.where(y_full == 0)[0]

        n_pos = min(len(pos_idx), 350)
        n_neg = min(len(neg_idx), 350)

        chosen_pos = np.random.choice(pos_idx, n_pos, replace=False) if len(pos_idx) > 0 else np.array([], dtype=int)
        chosen_neg = np.random.choice(neg_idx, n_neg, replace=False) if len(neg_idx) > 0 else np.array([], dtype=int)

        sampled_idx = np.concatenate([chosen_pos, chosen_neg])
        X_list.append(X_full[sampled_idx])
        y_list.append(y_full[sampled_idx])

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    print(f"   -> Extracted {len(X_train):,} multi-spectral pixel features across {len(t1_files)} training scenes.")
    print(f"   -> Class distribution: {np.sum(y_train==1):,} Deforested | {np.sum(y_train==0):,} Intact Canopy")
    
    rf.fit(X_train, y_train)
    out_path = ROOT_DIR / "weights" / "random_forest.joblib"
    rf.save(str(out_path))
    print(f"✅ Random Forest model successfully saved: {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_random_forest()
