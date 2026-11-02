"""
detection/area_calculation.py
==============================
Calculates forest area metrics from binary masks.

Area estimates use pixel_resolution_m (metres per pixel).
If no resolution is provided, pixel counts are reported instead — we never
pretend that raw pixel counts are hectares.

CO₂ sequestration estimate: ~200 tonnes CO₂ / hectare (tropical forest).
This is a literature-based approximation and is clearly labelled as such.
"""

from __future__ import annotations

import numpy as np

CO2_PER_HECTARE = 200.0   # tonnes CO₂ / ha  (tropical forest average, IPCC)
PIXELS_PER_HA_UNKNOWN = None  # sentinel value when no resolution available


def calculate_area_metrics(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    deforest_mask: np.ndarray,
    pixel_res_m: float | None = None,
    has_georeference: bool = False,
    risk_thresholds: tuple[float, float, float] = (1.0, 5.0, 10.0),
    carbon_factor: float = CO2_PER_HECTARE,
) -> dict:
    """
    Calculate area-based metrics from binary masks.

    Parameters
    ----------
    mask_t1          : (H, W) uint8  — binary forest mask, Time 1
    mask_t2          : (H, W) uint8  — binary forest mask, Time 2
    deforest_mask    : (H, W) uint8  — deforestation mask from change_detection
    pixel_res_m      : metres per pixel (e.g. 10 for Sentinel-2).
                       Pass None if resolution is unknown.
    has_georeference : True if resolution originates from verified GeoTIFF tags.
    risk_thresholds  : tuple of (low_thresh, mod_thresh, high_thresh) in percent.
    carbon_factor    : tonnes CO2 / hectare (IPCC tropical average default: 200.0).

    Returns
    -------
    dict with full area, loss, risk, and CO2 metrics.
    """
    f1_px = int((mask_t1 > 0).sum())
    f2_px = int((mask_t2 > 0).sum())
    df_px = int((deforest_mask > 0).sum())

    # Forest loss percentage formula:
    # ((Original Forest Area - Current Forest Area) / Original Forest Area) * 100
    # where loss pixels = df_px
    loss_pct = float(df_px / max(f1_px, 1) * 100)

    geo_available = pixel_res_m is not None and pixel_res_m > 0

    if geo_available:
        px_area_m2 = pixel_res_m ** 2
        px_area_ha = px_area_m2 / 10_000.0
        px_area_km2 = px_area_m2 / 1_000_000.0

        f1_m2  = f1_px  * px_area_m2
        f2_m2  = f2_px  * px_area_m2
        df_m2  = df_px  * px_area_m2

        f1_ha  = f1_px  * px_area_ha
        f2_ha  = f2_px  * px_area_ha
        df_ha  = df_px  * px_area_ha

        f1_km2 = f1_px  * px_area_km2
        f2_km2 = f2_px  * px_area_km2
        df_km2 = df_px  * px_area_km2

        co2    = df_ha  * carbon_factor
        if has_georeference:
            geo_desc = f"Verified from embedded GeoTIFF metadata ({pixel_res_m:.1f} m/px)"
            geo_verified = True
        else:
            geo_desc = f"Assumed from sensor preset ({pixel_res_m:.1f} m/px — uncalibrated optical imagery)"
            geo_verified = False

        note   = (
            f"Area estimated using spatial resolution {pixel_res_m:.1f} m/px [{geo_desc}]. "
            f"CO₂ estimate: ~{carbon_factor:.0f} t/ha (tropical forest, IPCC)."
        )
    else:
        f1_m2 = f2_m2 = df_m2 = None
        f1_ha = f2_ha = df_ha = None
        f1_km2 = f2_km2 = df_km2 = None
        co2 = None
        geo_desc = "Unavailable (pixel counts only)"
        geo_verified = False
        note = (
            "Area estimation requires image resolution/georeferencing. "
            "Only pixel counts are shown. Set pixel resolution to compute hectares."
        )

    # ---- Risk classification (project-defined, not an official standard) ----
    risk_level, risk_label = _classify_risk(loss_pct, thresholds=risk_thresholds)

    return {
        "pixel_res_m":      pixel_res_m,
        "geo_available":    geo_available,
        "geo_desc":         geo_desc,
        "geo_verified":     geo_verified,
        # pixel counts
        "forest_t1_pixels": f1_px,
        "forest_t2_pixels": f2_px,
        "deforest_pixels":  df_px,
        # square meters
        "forest_t1_m2":     f1_m2,
        "forest_t2_m2":     f2_m2,
        "deforest_m2":      df_m2,
        # geo areas (None if no resolution)
        "forest_t1_ha":     f1_ha,
        "forest_t2_ha":     f2_ha,
        "deforest_ha":      df_ha,
        "forest_t1_km2":    f1_km2,
        "forest_t2_km2":    f2_km2,
        "deforest_km2":     df_km2,
        # derived
        "forest_loss_pct":  loss_pct,
        "co2_loss_tonnes":  co2,
        "risk_level":       risk_level,
        "risk_label":       risk_label,
        "risk_thresholds":  risk_thresholds,
        "note":             note,
    }


def _classify_risk(
    loss_pct: float,
    thresholds: tuple[float, float, float] = (1.0, 5.0, 10.0),
) -> tuple[str, str]:
    """Project-defined risk classification. Not an official environmental standard."""
    th_low, th_mod, th_high = thresholds
    if loss_pct <= th_low:
        return "LOW",      "🟢 LOW"
    elif loss_pct <= th_mod:
        return "MODERATE", "🟡 MODERATE"
    elif loss_pct <= th_high:
        return "HIGH",     "🟠 HIGH"
    else:
        return "CRITICAL", "🔴 CRITICAL"

