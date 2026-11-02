import numpy as np

def calculate_ndvi(nir, red, eps=1e-7):
    """
    Normalized Difference Vegetation Index:
    NDVI = (NIR - Red) / (NIR + Red)
    Values range from -1.0 to +1.0:
    - Dense healthy canopy: 0.6 to 0.9
    - Shrub/sparse vegetation: 0.2 to 0.5
    - Bare soil / deforested: 0.05 to 0.2
    - Water/clouds: < 0.0
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1.0, 1.0)

def calculate_savi(nir, red, L=0.5, eps=1e-7):
    """
    Soil-Adjusted Vegetation Index:
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    Minimizes soil brightness influences in sparse canopy.
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    savi = ((nir - red) / (nir + red + L + eps)) * (1.0 + L)
    return np.clip(savi, -1.0, 1.0)

def compute_spectral_features(img_t1, img_t2):
    """
    Extracts multi-temporal spectral and difference features for pixel classification.
    Inputs are (H, W, 4) with channels [Red, Green, Blue, NIR].
    """
    t1_r, t1_g, t1_b = img_t1[:, :, 0] / 255.0, img_t1[:, :, 1] / 255.0, img_t1[:, :, 2] / 255.0
    t1_nir = (img_t1[:, :, 3] / 255.0) if img_t1.shape[2] > 3 else (t1_g * 1.5)

    t2_r, t2_g, t2_b = img_t2[:, :, 0] / 255.0, img_t2[:, :, 1] / 255.0, img_t2[:, :, 2] / 255.0
    t2_nir = (img_t2[:, :, 3] / 255.0) if img_t2.shape[2] > 3 else (t2_g * 1.5)

    ndvi_t1 = calculate_ndvi(t1_nir, t1_r)
    ndvi_t2 = calculate_ndvi(t2_nir, t2_r)
    delta_ndvi = ndvi_t1 - ndvi_t2  # High positive drop signifies vegetation loss

    savi_t1 = calculate_savi(t1_nir, t1_r)
    savi_t2 = calculate_savi(t2_nir, t2_r)
    delta_savi = savi_t1 - savi_t2

    # Color difference norms
    diff_rgb = np.sqrt((t2_r - t1_r)**2 + (t2_g - t1_g)**2 + (t2_b - t1_b)**2)
    diff_red = t2_r - t1_r  # Deforestation usually exposes reddish/brown soil

    return {
        "ndvi_t1": ndvi_t1,
        "ndvi_t2": ndvi_t2,
        "delta_ndvi": delta_ndvi,
        "savi_t1": savi_t1,
        "savi_t2": savi_t2,
        "delta_savi": delta_savi,
        "diff_rgb": diff_rgb,
        "diff_red": diff_red,
        "t1_rgb": np.stack([t1_r, t1_g, t1_b], axis=-1),
        "t2_rgb": np.stack([t2_r, t2_g, t2_b], axis=-1)
    }

def baseline_ndvi_threshold_detector(delta_ndvi, threshold=0.25):
    """
    Standard classical baseline method:
    Flags pixel as deforested if NDVI drops by more than `threshold`.
    """
    return (delta_ndvi >= threshold).astype(np.uint8)
