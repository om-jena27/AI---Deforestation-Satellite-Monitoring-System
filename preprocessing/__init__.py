"""Preprocessing package — image loading, harmonization, geospatial metadata."""
from .image_processing import (
    load_satellite_image,
    harmonize_image_pair,
    validate_image_pair,
    preprocess_for_segmentation,
    apply_noise_reduction,
    detect_clouds_rgb,
    compute_image_channel_stats,
    apply_display_stretch,
)
from .geospatial import (
    extract_spatial_resolution,
    calculate_ground_area,
    KNOWN_SENSOR_RESOLUTIONS,
)

__all__ = [
    "load_satellite_image",
    "harmonize_image_pair",
    "validate_image_pair",
    "preprocess_for_segmentation",
    "apply_noise_reduction",
    "detect_clouds_rgb",
    "compute_image_channel_stats",
    "apply_display_stretch",
    "extract_spatial_resolution",
    "calculate_ground_area",
    "KNOWN_SENSOR_RESOLUTIONS",
]

