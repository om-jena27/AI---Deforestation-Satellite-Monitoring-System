"""
detection/segmentation.py
=========================
Forest segmentation runner.

Given a raw satellite image array (H×W×C uint8) and a ModelManager instance,
returns:
    - forest_prob_map   : (H, W) float32  [0..1] — per-pixel forest probability
    - forest_mask       : (H, W) uint8    {0, 1} — thresholded binary mask
    - confidence_stats  : dict  {mean, max, min, threshold_used}

When the model is in DEMO mode (no trained checkpoint), a vegetation-index
proxy (green-channel prominence) is used and the result is clearly labelled.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from preprocessing.image_processing import preprocess_for_segmentation

# ---------------------------------------------------------------------------
# Vegetation-index proxy (used in DEMO mode)
# ---------------------------------------------------------------------------

def _ndvi_proxy_forest_probability(img_arr: np.ndarray) -> np.ndarray:
    """
    Approximate forest probability from a 3-channel RGB image.
    We use NIR-approximation from green channel prominence:
        proxy_ndvi = (G - R) / (G + R + 1e-6)
    Returns float32 (H, W) in [0, 1].
    """
    img_f = img_arr.astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0

    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    # Green Leaf Index — works well as proxy when no NIR band
    gli = (2 * g - r - b) / (2 * g + r + b + 1e-6)

    # Map [-1, 1] → [0, 1] with sigmoid-style stretching
    prob = 1.0 / (1.0 + np.exp(-5.0 * gli))  # sigmoid amplification

    # Apply gentle Gaussian smoothing to reduce noise
    prob = cv2.GaussianBlur(prob.astype(np.float32), (7, 7), 1.5)

    return np.clip(prob, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main segmentation function
# ---------------------------------------------------------------------------

def segment_forest(
    img_arr: np.ndarray,
    model_manager,
    target_size: tuple[int, int] = (256, 256),
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Segment forest pixels in a satellite image.

    Parameters
    ----------
    img_arr : np.ndarray (H, W, C) uint8 or float32
    model_manager : ModelManager instance from models/inference.py
    target_size : resize target for model input
    threshold : binary threshold applied to probability map

    Returns
    -------
    forest_prob_map : np.ndarray (H, W) float32  — raw probability map
                      resized back to original image dimensions
    forest_mask     : np.ndarray (H, W) uint8    — thresholded binary mask
    confidence_stats: dict with keys:
                        mean_confidence, max_confidence, min_confidence,
                        threshold_used, mode_label, is_trained
    """
    h_orig, w_orig = img_arr.shape[:2]

    # ------------------------------------------------------------------ #
    # 1.  Attempt real model inference
    # ------------------------------------------------------------------ #
    prob_map_small: np.ndarray | None = None

    if model_manager is not None and model_manager.is_trained:
        try:
            tensor = preprocess_for_segmentation(img_arr, target_size=target_size)
            prob_map_small = model_manager.predict_forest_probability(tensor)
        except Exception as exc:
            print(f"[segmentation] Model inference failed: {exc}. Falling back to proxy.")
            prob_map_small = None

    # ------------------------------------------------------------------ #
    # 2.  Fall back to vegetation-index proxy
    # ------------------------------------------------------------------ #
    mode_label: str
    is_trained: bool

    if prob_map_small is None:
        proxy_full = _ndvi_proxy_forest_probability(img_arr)
        # Downscale to target_size then back (consistent with model pipeline)
        proxy_small = cv2.resize(proxy_full, target_size, interpolation=cv2.INTER_AREA)
        prob_map_small = proxy_small
        mode_label = "DEMO — Vegetation Index Proxy (no trained model)"
        is_trained = False
    else:
        mode_label = getattr(model_manager, "mode_label", "REAL MODEL INFERENCE")
        is_trained = True

    # ------------------------------------------------------------------ #
    # 3.  Resize probability map back to original image dimensions
    # ------------------------------------------------------------------ #
    prob_map_full = cv2.resize(
        prob_map_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4.  Threshold → binary mask
    # ------------------------------------------------------------------ #
    forest_mask = (prob_map_full >= threshold).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # 5.  Detailed Forest Probability & Classification Statistics
    # ------------------------------------------------------------------ #
    f_px = int((forest_mask > 0).sum())
    nf_px = int((forest_mask == 0).sum())
    total_px = forest_mask.size

    confidence_stats = {
        "mean_confidence": float(np.mean(prob_map_full)),
        "max_confidence":  float(np.max(prob_map_full)),
        "min_confidence":  float(np.min(prob_map_full)),
        "mean_forest_prob": float(np.mean(prob_map_full)),
        "max_forest_prob":  float(np.max(prob_map_full)),
        "min_forest_prob":  float(np.min(prob_map_full)),
        "forest_pixels":    f_px,
        "non_forest_pixels": nf_px,
        "forest_pct":       float(f_px / max(total_px, 1) * 100.0),
        "non_forest_pct":   float(nf_px / max(total_px, 1) * 100.0),
        "threshold_used":  threshold,
        "mode_label":      mode_label,
        "is_trained":      is_trained,
    }

    return prob_map_full, forest_mask, confidence_stats
