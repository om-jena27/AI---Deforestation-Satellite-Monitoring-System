import numpy as np

def calculate_metrics(y_true, y_pred, pixel_resolution_meters=10.0):
    """
    Computes comprehensive remote sensing segmentation evaluation metrics:
    - Overall Accuracy (OA)
    - Precision
    - Recall (Sensitivity)
    - F1-Score / Dice Coefficient
    - IoU (Intersection over Union / Jaccard Index)
    - Specificity
    - Area Quantification (Hectares, sq km)
    - Estimated Biomass / CO2 Carbon Stock Loss
    """
    y_true_bin = (y_true.flatten() > 0).astype(np.uint8)
    y_pred_bin = (y_pred.flatten() > 0).astype(np.uint8)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Area calculation:
    # 1 pixel at Sentinel-2 10m resolution = 10m * 10m = 100 m²
    # 1 hectare = 10,000 m² = 100 pixels
    pixel_area_m2 = pixel_resolution_meters * pixel_resolution_meters
    predicted_loss_pixels = tp + fp
    true_loss_pixels = tp + fn
    
    predicted_area_ha = (predicted_loss_pixels * pixel_area_m2) / 10000.0
    true_area_ha = (true_loss_pixels * pixel_area_m2) / 10000.0
    predicted_area_km2 = predicted_area_ha / 100.0
    true_area_km2 = true_area_ha / 100.0

    # Carbon loss estimation (IPCC Tier-1 average tropical aboveground forest: ~150 tons Carbon/ha => ~550 tons CO2 eq/ha)
    co2_loss_tons = predicted_area_ha * 550.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_dice": round(float(f1), 4),
        "iou": round(float(iou), 4),
        "specificity": round(float(specificity), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "area_metrics": {
            "predicted_hectares": round(float(predicted_area_ha), 2),
            "true_hectares": round(float(true_area_ha), 2),
            "predicted_km2": round(float(predicted_area_km2), 4),
            "true_km2": round(float(true_area_km2), 4),
            "estimated_co2_loss_metric_tons": round(float(co2_loss_tons), 2)
        }
    }
