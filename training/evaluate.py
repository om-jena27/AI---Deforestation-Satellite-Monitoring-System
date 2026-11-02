"""
training/evaluate.py
====================
Standalone evaluation script for ForestUNet on independent test patches.
Can be executed via `python evaluate.py` or `python -m training.evaluate`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.unet import ForestUNet
from utils.metrics import compute_metrics

DATASET_DIR = ROOT_DIR / "dataset"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
WEIGHTS_DIR = ROOT_DIR / "weights"


def evaluate_test_set(
    checkpoint_path: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "unet_deforestation.pth"
        if not checkpoint_path.exists() and (WEIGHTS_DIR / "unet_deforestation.pth").exists():
            checkpoint_path = WEIGHTS_DIR / "unet_deforestation.pth"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        return {}

    test_t1 = sorted(glob.glob(str(DATASET_DIR / "test" / "t1" / "*.png")))
    test_t2 = sorted(glob.glob(str(DATASET_DIR / "test" / "t2" / "*.png")))
    test_masks = sorted(glob.glob(str(DATASET_DIR / "test" / "masks" / "*.png")))

    if not test_t1 or not test_masks:
        print(f"❌ Test dataset not found at {DATASET_DIR / 'test'}.")
        return {}

    print("=" * 65)
    print(f"🔬 Evaluating ForestUNet on Test Set ({len(test_t1)} image pairs)")
    print(f"   Checkpoint: {checkpoint_path.name}")
    print(f"   Device:     {device}")
    print("=" * 65)

    model = ForestUNet(in_channels=3, out_channels=1).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model_dict = model.state_dict()
    matching = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matching)
    model.load_state_dict(model_dict)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for f1_p, f2_p, m_p in zip(test_t1, test_t2, test_masks):
            t1 = np.array(Image.open(f1_p))[:, :, :3]
            t2 = np.array(Image.open(f2_p))[:, :, :3]
            gt_mask = np.array(Image.open(m_p))
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[:, :, 0]
            gt_deforest = (gt_mask > 127).astype(np.uint8)

            t1_t = torch.from_numpy(t1.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            t2_t = torch.from_numpy(t2.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            prob_t1 = model(t1_t).squeeze().cpu().numpy()
            prob_t2 = model(t2_t).squeeze().cpu().numpy()

            mask_t1 = (prob_t1 >= 0.5).astype(np.uint8)
            mask_t2 = (prob_t2 >= 0.5).astype(np.uint8)

            pred_deforest = ((mask_t1 == 1) & (mask_t2 == 0)).astype(np.uint8)

            all_preds.append(pred_deforest)
            all_targets.append(gt_deforest)

    pred_concat = np.concatenate([p.ravel() for p in all_preds])
    target_concat = np.concatenate([t.ravel() for t in all_targets])

    metrics = compute_metrics(pred_concat, target_concat)
    metrics["num_test_scenes"] = len(test_t1)
    metrics["total_pixels_evaluated"] = int(pred_concat.size)
    metrics["checkpoint_evaluated"] = checkpoint_path.name

    print("\n--- Quantitative Test Performance ---")
    print(f"  • Overall Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision:              {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  • Recall (Sensitivity):   {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  • F1 / Dice Score:        {metrics['f1']:.4f}")
    print(f"  • IoU (Jaccard Index):    {metrics['iou']:.4f}")

    out_file = output_json or (WEIGHTS_DIR / "evaluation_results.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Detailed evaluation saved to {out_file}")

    return metrics


if __name__ == "__main__":
    evaluate_test_set()
