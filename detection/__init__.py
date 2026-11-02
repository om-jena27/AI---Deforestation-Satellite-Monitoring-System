"""Detection package — segmentation, change detection, area calculation."""
from .segmentation import segment_forest
from .change_detection import detect_deforestation, find_deforestation_regions
from .area_calculation import calculate_area_metrics

__all__ = [
    "segment_forest",
    "detect_deforestation",
    "find_deforestation_regions",
    "calculate_area_metrics",
]
