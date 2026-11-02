"""
preprocessing/image_processing.py
===================================
Robust satellite image ingestion, format decoding, metadata extraction,
and pre-processing pipeline.

Supports:
- PNG, JPG/JPEG
- TIFF & GeoTIFF (via rasterio if available, with PIL tag extraction fallback)
- 16-bit to 8-bit dynamic range radiometric scaling
- Channel harmonization (grayscale -> 3ch, 4ch RGB+NIR, RGBA -> RGB)
- Satellite pair harmonization (dimension matching without distortion)
- Cloud & noise filtering (bilateral / median)
- Image alignment (ORB keypoint homography / affine)
- PyTorch tensor conversion for U-Net inference
"""

from __future__ import annotations

import io
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

try:
    import rasterio
    from rasterio.io import MemoryFile
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# 1. Image Loading & Metadata Extraction
# ---------------------------------------------------------------------------

def load_satellite_image(source: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Robustly loads a satellite image from a file path, file-like object,
    Streamlit UploadedFile, or numpy array.

    Supports PNG, JPG, JPEG, TIFF, and GeoTIFF formats.
    Extracts geospatial metadata when present (resolution, bounds, format).

    Returns:
        img_arr (np.ndarray): Decoded image array (H, W, C) in uint8 [0..255].
        metadata (dict): Image metadata dictionary.
    """
    metadata: dict[str, Any] = {
        "width": None,
        "height": None,
        "channels": None,
        "format": "Unknown",
        "size_kb": 0.0,
        "has_georeference": False,
        "pixel_resolution_m": None,
        "crs": None,
        "bounds": None,
        "bit_depth": "8-bit",
        "is_multispectral": False,
        "acquisition_date": None,
        "sensor_name": None,
        "cloud_cover_pct": 0.0,
        "band_names": [],
    }

    img_arr: np.ndarray | None = None
    raw_bytes: bytes | None = None

    # Case A: Streamlit UploadedFile or file-like object
    if hasattr(source, "read"):
        file_name = getattr(source, "name", "uploaded_image")
        metadata["format"] = file_name.split(".")[-1].upper()
        raw_bytes = source.read()
        source.seek(0)
        metadata["size_kb"] = round(len(raw_bytes) / 1024.0, 2)

        # Attempt rasterio first for GeoTIFF if installed
        if _HAS_RASTERIO and metadata["format"] in ("TIF", "TIFF", "GEOTIFF"):
            try:
                with MemoryFile(raw_bytes) as memfile:
                    with memfile.open() as src:
                        data = src.read()  # (C, H, W)
                        if data.ndim == 3:
                            img_arr = np.transpose(data, (1, 2, 0))
                        else:
                            img_arr = data
                        if src.res and src.res[0] > 0:
                            metadata["pixel_resolution_m"] = round(float(src.res[0]), 2)
                            metadata["has_georeference"] = True
                        if src.crs:
                            metadata["crs"] = str(src.crs)
                        if src.bounds:
                            metadata["bounds"] = {
                                "left": src.bounds.left,
                                "bottom": src.bounds.bottom,
                                "right": src.bounds.right,
                                "top": src.bounds.top,
                            }
            except Exception:
                img_arr = None

        # Fallback to PIL
        if img_arr is None:
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
                metadata["format"] = pil_img.format or metadata["format"]
                img_arr = _extract_pil_image_and_geotiff_tags(pil_img, metadata)
            except Exception:
                # OpenCV fallback
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img_arr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img_arr is not None and img_arr.ndim == 3 and img_arr.shape[2] == 3:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    # Case B: Local filesystem path
    elif isinstance(source, str) and os.path.exists(source):
        metadata["size_kb"] = round(os.path.getsize(source) / 1024.0, 2)
        metadata["format"] = source.split(".")[-1].upper()

        if _HAS_RASTERIO and metadata["format"] in ("TIF", "TIFF", "GEOTIFF"):
            try:
                with rasterio.open(source) as src:
                    data = src.read()
                    img_arr = np.transpose(data, (1, 2, 0)) if data.ndim == 3 else data
                    if src.res and src.res[0] > 0:
                        metadata["pixel_resolution_m"] = round(float(src.res[0]), 2)
                        metadata["has_georeference"] = True
                    if src.crs:
                        metadata["crs"] = str(src.crs)
                    if src.bounds:
                        metadata["bounds"] = {
                            "left": src.bounds.left,
                            "bottom": src.bounds.bottom,
                            "right": src.bounds.right,
                            "top": src.bounds.top,
                        }
            except Exception:
                img_arr = None

        if img_arr is None:
            try:
                pil_img = Image.open(source)
                metadata["format"] = pil_img.format or metadata["format"]
                img_arr = _extract_pil_image_and_geotiff_tags(pil_img, metadata)
            except Exception:
                img_arr = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                if img_arr is not None and img_arr.ndim == 3 and img_arr.shape[2] == 3:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    # Case C: Direct Numpy array
    elif isinstance(source, np.ndarray):
        img_arr = source.copy()
        metadata["format"] = "NUMPY"
        metadata["size_kb"] = round(img_arr.nbytes / 1024.0, 2)

    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    if img_arr is None:
        raise ValueError("Could not decode satellite image. Please verify file integrity.")

    # Radiometric normalization: 16-bit -> 8-bit
    if img_arr.dtype in (np.uint16, np.int16):
        metadata["bit_depth"] = "16-bit (normalized to 8-bit)"
        max_val = np.percentile(img_arr, 99.5)
        if max_val <= 0:
            max_val = 65535.0
        img_arr = np.clip((img_arr.astype(np.float32) / max_val) * 255.0, 0, 255).astype(np.uint8)
    elif img_arr.dtype in (np.float32, np.float64):
        metadata["bit_depth"] = "Float (normalized to 8-bit)"
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_arr = img_arr.clip(0, 255).astype(np.uint8)

    # Dimensions
    h, w = img_arr.shape[:2]
    c = img_arr.shape[2] if img_arr.ndim == 3 else 1
    metadata["width"] = w
    metadata["height"] = h
    metadata["channels"] = c
    metadata["is_multispectral"] = (c >= 4)

    # Cloud screening estimate
    try:
        rgb_preview = img_arr[:, :, :3] if img_arr.ndim == 3 else np.stack([img_arr] * 3, axis=-1)
        cloud_mask = detect_clouds_rgb(rgb_preview)
        metadata["cloud_cover_pct"] = round(float(np.sum(cloud_mask > 0)) / max(h * w, 1) * 100.0, 2)
    except Exception:
        metadata["cloud_cover_pct"] = 0.0

    return img_arr, metadata


def _extract_pil_image_and_geotiff_tags(pil_img: Image.Image, metadata: dict) -> np.ndarray:
    """
    Extracts raw image array and standard GeoTIFF tags from PIL:
    - Tag 33550: ModelPixelScaleTag [scaleX, scaleY, scaleZ]
    - Tag 33922: ModelTiepointTag [i, j, k, x, y, z]
    - Tag 34735: GeoKeyDirectoryTag (CRS / EPSG parsing)
    - Tag 34737: GeoAsciiParamsTag (Projection name)
    - Tag 306: DateTime (Satellite acquisition date/time)
    - Tag 305 / 270: Software / Sensor / Product description
    """
    tags = getattr(pil_img, "tag_v2", None) or getattr(pil_img, "tag", None)
    scale = None
    tiepoint = None

    if tags:
        # 1. ModelPixelScaleTag
        scale = tags.get(33550)
        if scale and len(scale) >= 1:
            try:
                res_x = float(scale[0])
                if res_x > 0:
                    metadata["pixel_resolution_m"] = round(res_x, 2)
                    metadata["has_georeference"] = True
            except Exception:
                pass

        # 2. ModelTiepointTag
        tiepoint = tags.get(33922)
        if tiepoint and len(tiepoint) >= 6:
            metadata["has_georeference"] = True

        # 3. Calculate spatial bounding box if both scale and tiepoint are present
        if scale and tiepoint and len(scale) >= 2 and len(tiepoint) >= 6:
            try:
                i, j, _, x, y, _ = tiepoint[:6]
                sx, sy = float(scale[0]), float(scale[1])
                w, h = pil_img.size
                left = float(x - i * sx)
                top = float(y + j * sy)
                right = float(left + w * sx)
                bottom = float(top - h * sy)
                metadata["bounds"] = {
                    "left": round(left, 4),
                    "bottom": round(bottom, 4),
                    "right": round(right, 4),
                    "top": round(top, 4),
                }
            except Exception:
                pass

        # 4. Parse CRS from GeoKeyDirectoryTag (Tag 34735)
        geokeys = tags.get(34735)
        if geokeys and len(geokeys) >= 4:
            try:
                num_keys = int(geokeys[3])
                for idx in range(num_keys):
                    pos = 4 + idx * 4
                    if pos + 4 <= len(geokeys):
                        k_id = int(geokeys[pos])
                        k_val = int(geokeys[pos + 3])
                        # Key 3072: ProjectedCSTypeGeoKey
                        if k_id == 3072 and k_val > 0:
                            metadata["crs"] = f"EPSG:{k_val}"
                            metadata["has_georeference"] = True
                            break
                        # Key 2048: GeographicTypeGeoKey
                        elif k_id == 2048 and k_val > 0 and metadata["crs"] is None:
                            metadata["crs"] = f"EPSG:{k_val}"
                            metadata["has_georeference"] = True
            except Exception:
                pass

        # 5. Fallback CRS from GeoAsciiParamsTag (Tag 34737)
        if metadata["crs"] is None and 34737 in tags:
            try:
                ascii_crs = str(tags[34737]).strip().rstrip("|").strip()
                if ascii_crs:
                    metadata["crs"] = ascii_crs
                    metadata["has_georeference"] = True
            except Exception:
                pass

        # 6. Acquisition Date/Time (Tag 306: DateTime)
        if 306 in tags:
            try:
                dt_str = str(tags[306]).strip()
                if dt_str:
                    # Convert 'YYYY:MM:DD HH:MM:SS' to 'YYYY-MM-DD HH:MM:SS'
                    parts = dt_str.split(" ")
                    if len(parts) >= 1:
                        date_part = parts[0].replace(":", "-")
                        time_part = parts[1] if len(parts) > 1 else ""
                        metadata["acquisition_date"] = f"{date_part} {time_part}".strip()
            except Exception:
                pass

        # 7. Sensor / Software description (Tag 305 or Tag 270)
        sensor_tag = tags.get(305) or tags.get(270)
        if sensor_tag:
            try:
                s_str = str(sensor_tag).strip()
                if s_str:
                    metadata["sensor_name"] = s_str
            except Exception:
                pass

    arr = np.array(pil_img)

    # Populate band names based on channel depth
    c = arr.shape[2] if arr.ndim == 3 else 1
    if c == 1:
        metadata["band_names"] = ["Panchromatic / Single-band"]
    elif c == 3:
        metadata["band_names"] = ["Band 1 (Red)", "Band 2 (Green)", "Band 3 (Blue)"]
    elif c == 4:
        metadata["band_names"] = ["Band 1 (Blue/Red)", "Band 2 (Green)", "Band 3 (Red/NIR)", "Band 4 (Near-Infrared/Alpha)"]
    elif c >= 6:
        metadata["band_names"] = [f"Band {i+1}" for i in range(c)]

    return arr


# ---------------------------------------------------------------------------
# 2. Image Pair Harmonization & Alignment
# ---------------------------------------------------------------------------

def harmonize_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    target_size: tuple[int, int] | None = None,
    apply_denoise: bool = False,
    align_pair: bool = False,
    preserve_multispectral: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensures Time 1 and Time 2 images have matching spatial dimensions and channels.
    Original spectral bands (including NIR/SWIR in 4+ band imagery) are strictly PRESERVED
    when preserve_multispectral=True, preventing silent data loss.

    Parameters:
        img1 (np.ndarray): Historical/Baseline (Time 1) image.
        img2 (np.ndarray): Current (Time 2) image.
        target_size: Optional fixed (W, H) tuple. If None, resizes img2 to match img1.
        apply_denoise: If True, applies gentle edge-preserving bilateral filtering.
        align_pair: If True, attempts feature-based image co-registration.
        preserve_multispectral: If True, preserves all channels (e.g. 4-band RGB+NIR);
                                if False, standardizes strictly to 3-band RGB.

    Returns:
        tuple[np.ndarray, np.ndarray]: Harmonized (img1, img2) arrays.
    """
    def prepare_channels(im: np.ndarray) -> np.ndarray:
        if im.ndim == 2:
            return np.stack([im, im, im], axis=-1)
        if im.shape[2] == 1:
            return np.concatenate([im, im, im], axis=-1)
        if not preserve_multispectral and im.shape[2] > 3:
            return im[:, :, :3]
        return im

    c1 = prepare_channels(img1.copy())
    c2 = prepare_channels(img2.copy())

    # If channel counts differ between the pair, harmonize to common minimum
    if c1.shape[2] != c2.shape[2]:
        min_c = min(c1.shape[2], c2.shape[2])
        c1 = c1[:, :, :min_c]
        c2 = c2[:, :, :min_c]

    h1, w1 = c1.shape[:2]
    h2, w2 = c2.shape[:2]

    # Target resizing
    if target_size is not None:
        tw, th = target_size
        c1 = cv2.resize(c1, (tw, th), interpolation=cv2.INTER_AREA)
        c2 = cv2.resize(c2, (tw, th), interpolation=cv2.INTER_AREA)
    elif (h1, w1) != (h2, w2):
        # Resize Time 2 to match Time 1 geometry
        c2 = cv2.resize(c2, (w1, h1), interpolation=cv2.INTER_AREA)

    # Optional co-registration alignment (features derived from RGB bands)
    if align_pair:
        aligned_rgb = align_satellite_pair(c1[:, :, :3], c2[:, :, :3])
        if c2.shape[2] > 3:
            c2[:, :, :3] = aligned_rgb
        else:
            c2 = aligned_rgb

    # Optional noise reduction
    if apply_denoise:
        c1 = apply_noise_reduction(c1)
        c2 = apply_noise_reduction(c2)

    return c1, c2

    return c1, c2


def align_satellite_pair(ref_img: np.ndarray, moving_img: np.ndarray) -> np.ndarray:
    """
    Aligns moving_img to ref_img using ORB feature detection and Homography.
    Falls back safely to moving_img if insufficient keypoints are found.
    """
    try:
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
        gray_mov = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray_ref, None)
        kp2, des2 = orb.detectAndCompute(gray_mov, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return moving_img

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:max(10, int(len(matches) * 0.25))]
        if len(good_matches) < 8:
            return moving_img

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return moving_img

        h, w = ref_img.shape[:2]
        aligned = cv2.warpPerspective(moving_img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return aligned
    except Exception:
        return moving_img


def apply_noise_reduction(img_rgb: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Applies edge-preserving noise filtering to satellite imagery."""
    if method == "bilateral":
        return cv2.bilateralFilter(img_rgb, d=7, sigmaColor=35, sigmaSpace=35)
    return cv2.medianBlur(img_rgb, 3)


def detect_clouds_rgb(img_rgb: np.ndarray, brightness_thresh: int = 215, saturation_thresh: int = 40) -> np.ndarray:
    """
    Detects high-reflectance cloud/haze artifacts in standard RGB satellite imagery.
    Returns binary mask (H, W) uint8 where 1 indicates cloud/haze.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    clouds = (val >= brightness_thresh) & (sat <= saturation_thresh)
    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clouds_clean = cv2.morphologyEx(clouds.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return clouds_clean


# ---------------------------------------------------------------------------
# 3. Model Inference Preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_segmentation(
    img_arr: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Prepares a single satellite image for U-Net forest segmentation.
    Performs resizing, normalization to [0.0, 1.0], and tensor conversion (1, 3, H, W).
    """
    tw, th = target_size
    resized = cv2.resize(img_arr[:, :, :3], (tw, th), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


# ---------------------------------------------------------------------------
# 4. Image Pair Compatibility & Geospatial Validation
# ---------------------------------------------------------------------------

def validate_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    meta1: dict[str, Any] | None = None,
    meta2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validates whether Time 1 and Time 2 images depict compatible dimensions,
    aspect ratios, and geographical areas.

    Prevents assuming arbitrary unaligned images from different locations or
    radically mismatched dimensions are directly comparable.
    """
    meta1 = meta1 or {}
    meta2 = meta2 or {}
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    ar1 = w1 / max(h1, 1)
    ar2 = w2 / max(h2, 1)
    ar_diff_pct = abs(ar1 - ar2) / max(ar1, 1e-6) * 100.0

    dimensions_match = (h1 == h2 and w1 == w2)
    aspect_ratio_compatible = ar_diff_pct <= 5.0  # within 5% tolerance

    warnings: list[str] = []
    errors: list[str] = []

    if not dimensions_match:
        if aspect_ratio_compatible:
            warnings.append(
                f"Dimension discrepancy ({w1}×{h1} vs {w2}×{h2} px). "
                "Aspect ratios match; Time 2 will be geometrically harmonized to Time 1."
            )
        else:
            warnings.append(
                f"Significant aspect ratio mismatch ({ar1:.2f} vs {ar2:.2f}, {ar_diff_pct:.1f}% difference). "
                "Resizing may distort spatial geometry and induce false change detections."
            )
            if ar_diff_pct > 50.0:
                errors.append(
                    f"Extreme aspect ratio disparity ({ar1:.2f} vs {ar2:.2f}, {ar_diff_pct:.1f}% difference). "
                    "Images depict incompatible spatial coverage and cannot be reliably aligned."
                )

    # Check bounds and spatial overlap if georeferenced
    geo_compatible: bool | None = None
    overlap_pct: float | None = None
    b1 = meta1.get("bounds")
    b2 = meta2.get("bounds")
    if meta1.get("has_georeference") and meta2.get("has_georeference") and b1 and b2:
        overlap_x = max(0.0, min(b1["right"], b2["right"]) - max(b1["left"], b2["left"]))
        overlap_y = max(0.0, min(b1["top"], b2["top"]) - max(b1["bottom"], b2["bottom"]))
        overlap_area = overlap_x * overlap_y
        area1 = max(1e-9, (b1["right"] - b1["left"]) * (b1["top"] - b1["bottom"]))
        area2 = max(1e-9, (b2["right"] - b2["left"]) * (b2["top"] - b2["bottom"]))
        overlap_pct = round((overlap_area / min(area1, area2)) * 100.0, 2)

        if overlap_pct > 0:
            geo_compatible = True
            if overlap_pct < 20.0:
                errors.append(
                    f"Insufficient spatial footprint overlap ({overlap_pct:.1f}%). "
                    "Images must share at least 20% geographical overlap for cross-temporal change detection."
                )
            elif overlap_pct < 50.0:
                warnings.append(
                    f"Partial geographical overlap ({overlap_pct:.1f}%). "
                    "Images do not share a common footprint; edge artifacts may occur."
                )
        else:
            geo_compatible = False
            errors.append(
                "Uploaded GeoTIFF files have non-overlapping spatial bounding boxes (0.0% overlap). "
                "They depict disjoint geographical regions and cannot be analyzed together."
            )
    elif not meta1.get("has_georeference") or not meta2.get("has_georeference"):
        warnings.append(
            "Geospatial CRS / GPS metadata unavailable. Ground alignment cannot be verified automatically."
        )

    # Check CRS consistency
    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")
    crs_matched = True
    if crs1 and crs2 and crs1 != crs2:
        crs_matched = False
        errors.append(
            f"Incompatible Coordinate Reference Systems (CRS): Time 1 is '{crs1}' while Time 2 is '{crs2}'. "
            "Cross-temporal change detection requires matching projection systems or prior reprojection."
        )

    # Check spatial resolution compatibility
    res1 = meta1.get("pixel_resolution_m")
    res2 = meta2.get("pixel_resolution_m")
    res_matched = True
    if res1 is not None and res2 is not None and res1 > 0 and res2 > 0:
        res_diff_pct = abs(res1 - res2) / max(res1, 1e-6) * 100.0
        if res_diff_pct > 25.0:
            res_matched = False
            errors.append(
                f"Incompatible Spatial Resolutions: Time 1 is {res1:.1f} m/px while Time 2 is {res2:.1f} m/px "
                f"({res_diff_pct:.1f}% difference). Sub-pixel change detection requires comparable ground resolution."
            )

    # Check spectral channels
    c1 = img1.shape[2] if img1.ndim == 3 else 1
    c2 = img2.shape[2] if img2.ndim == 3 else 1
    bands_matched = (c1 == c2)
    if c1 < 3 or c2 < 3:
        errors.append(
            f"Insufficient Spectral Bands: U-Net model requires at least 3 optical channels (Red, Green, Blue), "
            f"but Time 1 has {c1} band(s) and Time 2 has {c2} band(s)."
        )
    elif not bands_matched:
        warnings.append(
            f"Channel depth discrepancy: Time 1 has {c1} bands, while Time 2 has {c2} bands. "
            "Input arrays will be harmonized to their common band subset."
        )

    # Check cloud contamination (requires at least 3 channels)
    t1_cloud_pct = 0.0
    t2_cloud_pct = 0.0
    if c1 >= 3:
        rgb1 = img1[:, :, :3]
        t1_cloud_pct = round(float(np.sum(detect_clouds_rgb(rgb1))) / (h1 * w1) * 100.0, 2)
        if t1_cloud_pct > 10.0:
            warnings.append(
                f"High cloud contamination detected in Time 1 scene ({t1_cloud_pct:.1f}%). "
                "Opaque clouds or haze will obstruct underlying forest canopy observation."
            )
    if c2 >= 3:
        rgb2 = img2[:, :, :3]
        t2_cloud_pct = round(float(np.sum(detect_clouds_rgb(rgb2))) / (h2 * w2) * 100.0, 2)
        if t2_cloud_pct > 10.0:
            warnings.append(
                f"High cloud contamination detected in Time 2 scene ({t2_cloud_pct:.1f}%). "
                "Cloud cover and cloud shadows are prone to triggering false-positive deforestation alerts."
            )

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "dimensions_match": dimensions_match,
        "aspect_ratio_compatible": aspect_ratio_compatible,
        "aspect_ratio_diff_pct": round(ar_diff_pct, 2),
        "geo_overlap_verified": geo_compatible,
        "overlap_pct": overlap_pct,
        "crs_matched": crs_matched,
        "res_matched": res_matched,
        "t1_cloud_pct": t1_cloud_pct,
        "t2_cloud_pct": t2_cloud_pct,
        "bands_matched": bands_matched,
        "warnings": warnings,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 5. Image Diagnostics & Display Contrast Enhancements
# ---------------------------------------------------------------------------

def compute_image_channel_stats(img_arr: np.ndarray) -> dict[str, Any]:
    """
    Computes per-channel statistics (min, max, mean, std) and sample representative pixels.
    Enables deep inspection of raw optical satellite imagery and detection of narrow dynamic ranges.
    """
    rgb = img_arr[:, :, :3]
    h, w, c = rgb.shape

    sample_coords = [
        (0, 0),
        (h // 4, w // 4),
        (h // 2, w // 2),
        (3 * h // 4, 3 * w // 4),
        (h - 1, w - 1),
    ]
    samples = [{"coord": f"({x},{y})", "rgb": [int(v) for v in rgb[y, x]]} for x, y in sample_coords]

    stats = {
        "dimensions": f"{w} × {h} × {c}",
        "width": w,
        "height": h,
        "channels": c,
        "dtype": str(img_arr.dtype),
        "r_min": int(rgb[:, :, 0].min()),
        "r_max": int(rgb[:, :, 0].max()),
        "r_mean": float(np.mean(rgb[:, :, 0])),
        "r_std": float(np.std(rgb[:, :, 0])),
        "g_min": int(rgb[:, :, 1].min()),
        "g_max": int(rgb[:, :, 1].max()),
        "g_mean": float(np.mean(rgb[:, :, 1])),
        "g_std": float(np.std(rgb[:, :, 1])),
        "b_min": int(rgb[:, :, 2].min()),
        "b_max": int(rgb[:, :, 2].max()),
        "b_mean": float(np.mean(rgb[:, :, 2])),
        "b_std": float(np.std(rgb[:, :, 2])),
        "overall_min": int(rgb.min()),
        "overall_max": int(rgb.max()),
        "samples": samples,
        "is_low_reflectance": bool(rgb.max() <= 80),
    }
    return stats


def apply_display_stretch(
    img_rgb: np.ndarray,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0,
) -> np.ndarray:
    """
    Applies standard remote sensing percentile contrast stretching for DISPLAY PURPOSES ONLY.
    The original raw numpy array remains strictly untouched for deep learning model inference.
    """
    stretched = np.zeros_like(img_rgb, dtype=np.uint8)
    for ch in range(3):
        band = img_rgb[:, :, ch].astype(np.float32)
        p_low = np.percentile(band, lower_pct)
        p_high = np.percentile(band, upper_pct)
        if p_high > p_low:
            scaled = np.clip((band - p_low) / (p_high - p_low) * 255.0, 0, 255)
            stretched[:, :, ch] = scaled.astype(np.uint8)
        else:
            stretched[:, :, ch] = img_rgb[:, :, ch]
    return stretched


