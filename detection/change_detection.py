"""
detection/change_detection.py
==============================
Pixel-level deforestation detection from two binary forest masks.

Key rule:
    deforestation = FOREST in Time-1  AND  NON-FOREST in Time-2
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core change detection
# ---------------------------------------------------------------------------

def detect_deforestation(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Compare two binary forest masks and compute the deforestation mask.

    Parameters
    ----------
    mask_t1 : (H, W) uint8  — binary forest mask for Time 1 (1=forest, 0=non-forest)
    mask_t2 : (H, W) uint8  — binary forest mask for Time 2

    Returns
    -------
    deforest_mask : (H, W) uint8  — 1 where deforestation detected
    stats : dict with:
        total_pixels, forest_t1_pixels, forest_t2_pixels,
        forest_retained_pixels, forest_loss_pixels,
        new_vegetation_pixels, no_change_pixels,
        forest_loss_pct, forest_retained_pct, new_veg_pct
    """
    assert mask_t1.shape == mask_t2.shape, (
        f"Mask shape mismatch: {mask_t1.shape} vs {mask_t2.shape}"
    )

    m1 = mask_t1.astype(bool)
    m2 = mask_t2.astype(bool)

    deforest   = m1 & ~m2          # was forest, now not
    retained   = m1 & m2           # still forest
    new_veg    = ~m1 & m2          # was not forest, now is (regrowth / new planting)
    no_change_nf = ~m1 & ~m2       # neither time was forest

    total = mask_t1.size
    f_t1  = int(m1.sum())
    f_t2  = int(m2.sum())

    deforest_mask = deforest.astype(np.uint8)

    nf_t1 = total - f_t1
    nf_t2 = total - f_t2
    df_count = int(deforest.sum())

    stats = {
        "total_pixels":          total,
        "forest_t1_pixels":      f_t1,
        "non_forest_t1_pixels":  nf_t1,
        "forest_t1_pct":         float(f_t1 / max(total, 1) * 100),
        "non_forest_t1_pct":     float(nf_t1 / max(total, 1) * 100),
        "forest_t2_pixels":      f_t2,
        "non_forest_t2_pixels":  nf_t2,
        "forest_t2_pct":         float(f_t2 / max(total, 1) * 100),
        "non_forest_t2_pct":     float(nf_t2 / max(total, 1) * 100),
        "forest_loss_pixels":    df_count,
        "deforest_pixel_pct":    float(df_count / max(total, 1) * 100),
        "forest_retained_pixels":int(retained.sum()),
        "new_vegetation_pixels": int(new_veg.sum()),
        "no_change_pixels":      int(no_change_nf.sum()),
        "forest_loss_pct":       float(df_count / max(f_t1, 1) * 100),
        "forest_retained_pct":   float(retained.sum() / max(f_t1, 1) * 100),
        "new_veg_pct":           float(new_veg.sum() / max(total, 1) * 100),
    }

    return deforest_mask, stats


# ---------------------------------------------------------------------------
# Connected-component region analysis
# ---------------------------------------------------------------------------

def find_deforestation_regions(
    deforest_mask: np.ndarray,
    confidence_map: np.ndarray | None = None,
    pixel_res_m: float | None = None,
    min_area_pixels: int = 50,
) -> list[dict]:
    """
    Find individual deforestation regions using connected components.

    Parameters
    ----------
    deforest_mask : (H, W) uint8
    confidence_map : Optional (H, W) float32 probability/confidence map
    pixel_res_m   : Optional spatial resolution (meters/pixel)
    min_area_pixels : minimum region size (smaller regions are treated as noise)

    Returns
    -------
    list of dicts, each with:
        id, area_pixels, area_ha, area_str, confidence_pct,
        bbox (x, y, w, h), centroid (cx, cy)
    """
    # cv2.connectedComponentsWithStats uses 8-connectivity
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        deforest_mask.astype(np.uint8), connectivity=8
    )

    regions: list[dict] = []
    for i in range(1, num_labels):  # skip background (label 0)
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < min_area_pixels:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[i, 0])
        cy = float(centroids[i, 1])

        # Region confidence calculation
        if confidence_map is not None:
            region_mask = (labels == i)
            conf_vals = confidence_map[region_mask]
            conf_score = float(np.mean(conf_vals)) if len(conf_vals) > 0 else 0.0
            conf_pct = round(conf_score * 100.0, 1)
        else:
            conf_pct = None

        # Area string formatting
        if pixel_res_m is not None and pixel_res_m > 0:
            area_ha = round((area_px * (pixel_res_m ** 2)) / 10_000.0, 3)
            area_str = f"{area_ha:.2f} ha"
        else:
            area_ha = None
            area_str = f"{area_px:,} px"

        regions.append({
            "id":             i,
            "area_pixels":    area_px,
            "area_ha":        area_ha,
            "area_str":       area_str,
            "confidence_pct": conf_pct,
            "bbox":           {"x": x, "y": y, "w": w, "h": h},
            "centroid":       {"cx": cx, "cy": cy},
        })

    # Sort largest first
    regions.sort(key=lambda r: r["area_pixels"], reverse=True)

    # Re-index cleanly
    for idx, r in enumerate(regions, start=1):
        r["id"] = idx

    return regions

