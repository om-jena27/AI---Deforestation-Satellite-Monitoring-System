"""
preprocessing/geospatial.py
===========================
Geospatial resolution and ground sampling distance (GSD) utilities.

Ensures real-world ground area calculations (hectares, km², m²) are only
performed when calibrated spatial resolution or georeferencing is available.
"""

from __future__ import annotations

from typing import Any

# Known earth observation sensor ground resolutions (GSD in meters/pixel)
KNOWN_SENSOR_RESOLUTIONS: dict[str, float] = {
    "Sentinel-2 (MSI VNIR)": 10.0,
    "Landsat 8/9 (OLI)": 30.0,
    "PlanetScope / Dove": 3.0,
    "SkySat (High-Res)": 0.5,
    "MODIS (Vegetation)": 250.0,
    "Custom / User-Defined": 10.0,
}


def extract_spatial_resolution(
    metadata: dict[str, Any] | None,
    user_resolution: float | None = None,
) -> tuple[float | None, bool, str]:
    """
    Determines the spatial resolution (meters per pixel).

    Priority:
    1. Georeferenced image metadata (GeoTIFF tags, rasterio GSD)
    2. User-provided resolution (if > 0)
    3. None (resolution unknown - pixel counts only)

    Returns:
        pixel_res_m (float | None): Resolution in meters/pixel, or None.
        is_georeferenced (bool): True if derived from verified metadata.
        source_description (str): Human-readable provenance label.
    """
    if metadata and metadata.get("pixel_resolution_m") is not None:
        try:
            val = float(metadata["pixel_resolution_m"])
            if val > 0:
                return val, True, f"Embedded GeoTIFF metadata ({val:.1f} m/px)"
        except (ValueError, TypeError):
            pass

    if user_resolution is not None and user_resolution > 0:
        return float(user_resolution), False, f"User-specified resolution ({user_resolution:.1f} m/px)"

    return None, False, "Resolution not specified (pixel counts only)"


def calculate_ground_area(
    pixel_count: int,
    pixel_res_m: float | None,
) -> dict[str, float | None]:
    """
    Converts pixel count to real-world ground area units.
    If pixel_res_m is None or <= 0, area units return None.
    """
    if pixel_res_m is None or pixel_res_m <= 0:
        return {
            "sq_meters": None,
            "hectares": None,
            "sq_km": None,
        }

    area_m2 = float(pixel_count) * (pixel_res_m ** 2)
    area_ha = area_m2 / 10_000.0
    area_km2 = area_m2 / 1_000_000.0

    return {
        "sq_meters": round(area_m2, 2),
        "hectares": round(area_ha, 4),
        "sq_km": round(area_km2, 6),
    }
