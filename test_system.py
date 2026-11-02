"""
test_system.py
==============
Comprehensive end-to-end integration test for the AI Deforestation Monitoring System.
Verifies all 10 core pipeline stages:
1. ModelManager Checkpoint Detection (Real Inference Mode)
2. Satellite Image Loading & Metadata Decoding
3. Image Harmonization & Radiometric Normalization
4. Deep Learning U-Net Forest Canopy Segmentation (T1 & T2)
5. Temporal Change Detection & Deforestation Mask
6. Connected-Component Region Analysis with Confidence Mapping
7. Geospatial Ground Area (Hectares, km²) & Risk Classification
8. Interactive Plotly Detection Map Generation
9. Professional PDF Report Generation (ReportLab)
10. Demo Mode Toggle & Vegetation Proxy Verification
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import numpy as np
from PIL import Image
import torch

from models.inference import ModelManager, get_model_manager
from models.unet import ForestUNet
from preprocessing.image_processing import load_satellite_image, harmonize_image_pair, preprocess_for_segmentation
from preprocessing.geospatial import extract_spatial_resolution, calculate_ground_area
from detection.segmentation import segment_forest
from detection.change_detection import detect_deforestation, find_deforestation_regions
from detection.area_calculation import calculate_area_metrics
from utils.visualization import create_deforestation_overlay, create_interactive_detection_map
from utils.report import generate_pdf_report


def run_pipeline_verification():
    print("=" * 65)
    print("🛰️ AI Deforestation System — Full Pipeline Verification")
    print("=" * 65)

    # 1. Model Loading
    print("\n[Step 1] Initializing ModelManager & Checkpoint Detection...")
    mm = ModelManager(force_demo=False)
    print(f"   -> Mode: {mm.mode_label}")
    print(f"   -> Trained Checkpoint Loaded: {mm.is_trained}")
    print(f"   -> Device: {mm.device}")
    assert mm.is_trained, "Error: Trained U-Net checkpoint not loaded! Run train.py first."

    # 2. Image Ingestion
    print("\n[Step 2] Testing Satellite Image Loading & Metadata Extraction...")
    sample_t1 = "dataset/sample_pairs/t1/01_amazon_fishbone_t1.png"
    sample_t2 = "dataset/sample_pairs/t2/01_amazon_fishbone_t2.png"

    img_t1_raw, meta_t1 = load_satellite_image(sample_t1)
    img_t2_raw, meta_t2 = load_satellite_image(sample_t2)

    print(f"   -> T1: {meta_t1['width']}x{meta_t1['height']} px | {meta_t1['channels']} channels | {meta_t1['format']}")
    print(f"   -> T2: {meta_t2['width']}x{meta_t2['height']} px | {meta_t2['channels']} channels | {meta_t2['format']}")

    # 3. Harmonization
    print("\n[Step 3] Testing Image Harmonization & Normalization...")
    t1_harm, t2_harm = harmonize_image_pair(img_t1_raw, img_t2_raw)
    assert t1_harm.shape == t2_harm.shape, "Shape mismatch after harmonization!"
    print(f"   -> Harmonized Shape: {t1_harm.shape} (uint8)")

    # 4. Deep Learning Segmentation
    print("\n[Step 4] Running PyTorch U-Net Forest Canopy Segmentation...")
    prob_t1, mask_t1, conf_t1 = segment_forest(t1_harm, mm, threshold=0.5)
    prob_t2, mask_t2, conf_t2 = segment_forest(t2_harm, mm, threshold=0.5)

    forest_t1_pct = (mask_t1 > 0).mean() * 100
    forest_t2_pct = (mask_t2 > 0).mean() * 100
    print(f"   -> T1 Forest Canopy Cover: {forest_t1_pct:.2f}% (Mean Confidence: {conf_t1['mean_confidence']:.3f})")
    print(f"   -> T2 Forest Canopy Cover: {forest_t2_pct:.2f}% (Mean Confidence: {conf_t2['mean_confidence']:.3f})")

    # 5. Temporal Change Detection
    print("\n[Step 5] Running Change Detection (Deforestation Condition)...")
    deforest_mask, change_stats = detect_deforestation(mask_t1, mask_t2)
    loss_px = change_stats["forest_loss_pixels"]
    loss_pct = change_stats["forest_loss_pct"]
    print(f"   -> Detected Deforestation Pixels: {loss_px:,} px ({loss_pct:.2f}% loss)")
    print(f"   -> Forest Retained Pixels:       {change_stats['forest_retained_pixels']:,} px")

    # 6. Connected Component Analysis
    print("\n[Step 6] Identifying Individual Deforestation Zones...")
    prob_deforest = prob_t1 * (1.0 - prob_t2)
    regions = find_deforestation_regions(
        deforest_mask, confidence_map=prob_deforest, pixel_res_m=10.0, min_area_pixels=30
    )
    print(f"   -> Identified {len(regions)} disturbance zones above 30 px threshold.")
    if regions:
        top_zone = regions[0]
        print(f"   -> Zone #{top_zone['id']:02d}: Area = {top_zone['area_str']} | Confidence = {top_zone['confidence_pct']}%")

    # 7. Geospatial & Risk Quantification
    print("\n[Step 7] Calculating Real-World Geospatial Area & Risk Level...")
    area_metrics = calculate_area_metrics(
        mask_t1, mask_t2, deforest_mask, pixel_res_m=10.0, carbon_factor=200.0
    )
    print(f"   -> Deforested Area:   {area_metrics['deforest_ha']:,.2f} hectares ({area_metrics['deforest_km2']:.4f} km²)")
    print(f"   -> Canopy Loss Ratio: {area_metrics['forest_loss_pct']:.2f}%")
    print(f"   -> Risk Level:        {area_metrics['risk_label']} (Project-defined scale)")
    print(f"   -> Carbon Released:   ~{area_metrics['co2_loss_tonnes']:,.1f} tonnes CO2")

    # 8. Interactive Plotly Map
    print("\n[Step 8] Generating Interactive Plotly Detection Map...")
    fig = create_interactive_detection_map(t2_harm, deforest_mask, regions=regions, pixel_res_m=10.0)
    assert fig is not None, "Failed to build Plotly figure!"
    print("   -> Interactive Plotly Map constructed successfully.")

    # 9. PDF Report Generation
    print("\n[Step 9] Generating Professional PDF Audit Report (ReportLab)...")
    images_dict = {
        "t1_img": t1_harm,
        "t2_img": t2_harm,
        "forest_mask_t1": (mask_t1 * 255).astype(np.uint8),
        "forest_mask_t2": (mask_t2 * 255).astype(np.uint8),
        "deforest_overlay": create_deforestation_overlay(t2_harm, deforest_mask, opacity=0.65),
        "deforest_mask_display": (deforest_mask * 255).astype(np.uint8),
    }
    pdf_bytes = generate_pdf_report(area_metrics, images_dict)
    assert len(pdf_bytes) > 5000, "PDF generation output is unexpectedly small!"
    print(f"   -> Generated valid PDF report: {len(pdf_bytes):,} bytes.")

    # 10. Demo Mode Verification
    print("\n[Step 10] Testing Demo Mode Fallback Pipeline...")
    mm_demo = ModelManager(force_demo=True)
    _, _, conf_demo = segment_forest(t1_harm, mm_demo)
    assert not conf_demo["is_trained"], "Demo mode failed to reflect untrained status!"
    print(f"   -> Demo mode verified: '{conf_demo['mode_label']}'")

    # 11. Error Handling (Missing Files & Corrupted Images)
    print("\n[Step 11] Testing Error Handling (Missing & Corrupted Files)...")
    import io
    try:
        load_satellite_image("non_existent_file_xyz.png")
        assert False, "Should have raised error for missing file!"
    except (ValueError, FileNotFoundError):
        print("   -> Missing file correctly caught and handled.")

    try:
        load_satellite_image(io.BytesIO(b"MALFORMED_HEADER_NOT_A_VALID_IMAGE_DATA"))
        assert False, "Should have raised error for corrupted stream!"
    except ValueError:
        print("   -> Corrupted image stream correctly detected.")

    # 12. Mismatched Dimensions & Image Compatibility Validation
    print("\n[Step 12] Testing Image Pair Validation & Harmonization...")
    from preprocessing.image_processing import validate_image_pair
    img_diff1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img_diff2 = np.zeros((512, 512, 3), dtype=np.uint8)
    val_res = validate_image_pair(img_diff1, img_diff2)
    assert not val_res["dimensions_match"]
    assert val_res["aspect_ratio_compatible"]
    harm1, harm2 = harmonize_image_pair(img_diff1, img_diff2)
    assert harm1.shape == harm2.shape == (256, 256, 3)
    print("   -> Dimension mismatch harmonized without crash.")

    # 13. Exact Math & Area Calculation Verification
    print("\n[Step 13] Verifying Exact Math: 2,621.44 ha - 2,516.93 ha = 104.51 ha (3.99%)...")
    m1_full = np.ones((512, 512), dtype=np.uint8)
    m2_full = np.ones((512, 512), dtype=np.uint8)
    m2_full.ravel()[:10_451] = 0  # 10,451 cleared pixels
    df_m, _ = detect_deforestation(m1_full, m2_full)
    am_calc = calculate_area_metrics(m1_full, m2_full, df_m, pixel_res_m=10.0)
    assert am_calc["forest_t1_ha"] == 2621.44
    assert am_calc["forest_t2_ha"] == 2516.93
    assert round(am_calc["forest_t1_ha"] - am_calc["forest_t2_ha"], 2) == 104.51
    assert round(am_calc["deforest_ha"], 2) == 104.51
    assert round(am_calc["forest_loss_pct"], 2) == 3.99
    print(f"   -> Math verified: {am_calc['forest_t1_ha']:,.2f} ha - {am_calc['forest_t2_ha']:,.2f} ha = {am_calc['deforest_ha']:,.2f} ha ({am_calc['forest_loss_pct']:.2f}% loss).")

    # 14. Non-Fabricated Benchmark Verification
    print("\n[Step 14] Verifying Honest Test & Benchmark Results...")
    import json
    with open("weights/benchmark_results.json", "r") as f_b:
        bench_data = json.load(f_b)
    assert bench_data.get("U-Net (with Augmentation)") == "Not evaluated"
    assert bench_data.get("DeepLabV3+") == "Not evaluated"
    assert "Accuracy" in bench_data["U-Net (ForestUNet)"]
    print("   -> Benchmark results verified: strictly honest 'Not evaluated' for non-evaluated architectures.")

    # 15. Real Georeferenced GeoTIFF Pipeline Verification
    print("\n[Step 15] Verifying Real Georeferenced GeoTIFF Workflow...")
    real_t1 = "dataset/real_satellite_samples/t1/sentinel2_rondonia_20200815_t1.tif"
    real_t2 = "dataset/real_satellite_samples/t2/sentinel2_rondonia_20230820_t2.tif"
    r_arr1, r_meta1 = load_satellite_image(real_t1)
    r_arr2, r_meta2 = load_satellite_image(real_t2)
    assert r_meta1["has_georeference"] is True
    assert r_meta1["crs"] == "EPSG:32720"
    assert r_meta1["pixel_resolution_m"] == 10.0
    assert r_meta1["channels"] == 4
    assert r_meta1["is_multispectral"] is True
    assert r_meta1["acquisition_date"] == "2020-08-15 14:22:10"
    r_val = validate_image_pair(r_arr1, r_arr2, r_meta1, r_meta2)
    assert r_val["is_valid"] is True
    assert r_val["overlap_pct"] == 100.0
    print(f"   -> Real GeoTIFF Verified: CRS={r_meta1['crs']} | GSD={r_meta1['pixel_resolution_m']}m | Overlap={r_val['overlap_pct']}% | Acquisition={r_meta1['acquisition_date']}")

    # 16. Model Domain Compatibility & Retraining Pipeline
    print("\n[Step 16] Verifying Domain Compatibility & Real Training Pipeline...")
    compat = mm.check_domain_compatibility({"is_synthetic": False})
    assert compat["has_domain_mismatch"] is True
    assert "train_real" in compat["recommended_workflow"]
    from training.train_real import RealSatelliteDataset
    real_ds = RealSatelliteDataset([(real_t1, "dataset/real_satellite_samples/masks/sentinel2_rondonia_reference_mask.tif")], in_channels=4)
    t4_sample, m_sample = real_ds[0]
    assert t4_sample.shape == (4, 256, 256)
    print("   -> Domain mismatch correctly flagged; Real retraining dataset pipeline verified.")

    print("\n" + "=" * 65)
    print("🎉 ALL 16 PIPELINE & INTEGRITY CHECKS VERIFIED AND PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline_verification()
