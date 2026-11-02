import os
import json
import numpy as np
from PIL import Image

# Directories
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR)
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
SAMPLES_DIR = os.path.join(DATASET_DIR, "sample_pairs")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, SAMPLES_DIR]:
    for sub in ["t1", "t2", "masks"]:
        os.makedirs(os.path.join(d, sub), exist_ok=True)

np.random.seed(42)

def generate_satellite_scene(scene_type="fishbone", img_size=256):
    """
    Simulates high-fidelity Sentinel-2 multispectral satellite scene:
    Channels: Red, Green, Blue, Near-Infrared (NIR)
    
    scene_type options:
    - 'fishbone': Road corridor logging typical of Rondônia, Amazon
    - 'clearcut': Rectangular & irregular agricultural patches
    - 'selective_logging': Diffuse scattered forest thinning
    - 'wildfire': Irregular burn scars
    - 'intact_forest': Control scene with high canopy density (no change)
    """
    # 1. Base canopy texture (Perlin-like smooth noise + high frequency texture)
    y, x = np.mgrid[0:img_size, 0:img_size] / float(img_size)
    
    # Base forest terrain variation
    terrain = np.sin(3 * np.pi * x) * np.cos(3 * np.pi * y) * 0.15 + 0.5
    
    # Healthy forest spectral signatures (Sentinel-2 typical reflectance):
    # NIR: High (0.6 - 0.85), Red: Low (0.04 - 0.12), Green: Moderate (0.08 - 0.18), Blue: Low (0.03 - 0.08)
    forest_nir = np.clip(0.65 + 0.15 * terrain + np.random.normal(0, 0.03, (img_size, img_size)), 0.45, 0.95)
    forest_red = np.clip(0.08 + 0.04 * (1 - terrain) + np.random.normal(0, 0.015, (img_size, img_size)), 0.03, 0.20)
    forest_green = np.clip(0.14 + 0.05 * terrain + np.random.normal(0, 0.02, (img_size, img_size)), 0.06, 0.30)
    forest_blue = np.clip(0.05 + 0.02 * (1 - terrain) + np.random.normal(0, 0.01, (img_size, img_size)), 0.02, 0.15)
    
    # T1: Baseline dense forest
    t1_rgb = np.stack([forest_red, forest_green, forest_blue], axis=-1)
    t1_nir = forest_nir

    # T2 Initialization (copied from T1)
    t2_rgb = t1_rgb.copy()
    t2_nir = t1_nir.copy()
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Deforested spectral signatures (exposed soil, slash, dry pasture):
    # NIR: Low (0.15 - 0.30), Red: High (0.25 - 0.45), Green: Moderate (0.18 - 0.28), Blue: Moderate (0.12 - 0.22)
    def apply_deforestation(patch_mask):
        soil_red = np.clip(0.35 + np.random.normal(0, 0.04, (img_size, img_size)), 0.20, 0.55)
        soil_green = np.clip(0.24 + np.random.normal(0, 0.03, (img_size, img_size)), 0.15, 0.40)
        soil_blue = np.clip(0.16 + np.random.normal(0, 0.02, (img_size, img_size)), 0.08, 0.30)
        soil_nir = np.clip(0.22 + np.random.normal(0, 0.03, (img_size, img_size)), 0.10, 0.35)
        
        t2_rgb[patch_mask == 1, 0] = soil_red[patch_mask == 1]
        t2_rgb[patch_mask == 1, 1] = soil_green[patch_mask == 1]
        t2_rgb[patch_mask == 1, 2] = soil_blue[patch_mask == 1]
        t2_nir[patch_mask == 1] = soil_nir[patch_mask == 1]
        mask[patch_mask == 1] = 1

    if scene_type == "fishbone":
        # Central logging road with perpendicular lateral spurs
        m = np.zeros((img_size, img_size), dtype=np.uint8)
        # Main road
        road_center = np.random.randint(80, 170)
        road_width = np.random.randint(6, 12)
        m[:, max(0, road_center - road_width):min(img_size, road_center + road_width)] = 1
        # Fishbone side clears
        num_spurs = np.random.randint(3, 7)
        for i in range(num_spurs):
            y_pos = int((i + 1) * img_size / (num_spurs + 1)) + np.random.randint(-10, 10)
            spur_len = np.random.randint(50, 95)
            spur_height = np.random.randint(18, 35)
            # Left spur
            x_start = max(0, road_center - spur_len)
            m[max(0, y_pos - spur_height//2):min(img_size, y_pos + spur_height//2), x_start:road_center] = 1
            # Right spur
            x_end = min(img_size, road_center + spur_len)
            m[max(0, y_pos - spur_height//2):min(img_size, y_pos + spur_height//2), road_center:x_end] = 1
        apply_deforestation(m)

    elif scene_type == "clearcut":
        # Large rectangular/trapezoid agricultural clearings
        m = np.zeros((img_size, img_size), dtype=np.uint8)
        num_blocks = np.random.randint(2, 5)
        for _ in range(num_blocks):
            rx1 = np.random.randint(20, img_size - 90)
            ry1 = np.random.randint(20, img_size - 90)
            rw = np.random.randint(45, 100)
            rh = np.random.randint(45, 100)
            m[ry1:min(img_size, ry1+rh), rx1:min(img_size, rx1+rw)] = 1
        apply_deforestation(m)

    elif scene_type == "selective_logging":
        # Circular / irregular pockets of canopy removal
        m = np.zeros((img_size, img_size), dtype=np.uint8)
        num_pockets = np.random.randint(8, 20)
        for _ in range(num_pockets):
            cx, cy = np.random.randint(20, img_size-20), np.random.randint(20, img_size-20)
            rad = np.random.randint(10, 25)
            dist_sq = (x * img_size - cx)**2 + (y * img_size - cy)**2
            m[dist_sq <= rad**2] = 1
        apply_deforestation(m)

    elif scene_type == "wildfire":
        # Irregular organic burn patch
        m = np.zeros((img_size, img_size), dtype=np.uint8)
        cx, cy = np.random.randint(img_size//3, 2*img_size//3), np.random.randint(img_size//3, 2*img_size//3)
        dist = np.sqrt((x * img_size - cx)**2 + (y * img_size - cy)**2)
        noise = np.random.normal(0, 15, (img_size, img_size))
        m[(dist + noise) < 65] = 1
        apply_deforestation(m)

    elif scene_type == "intact_forest":
        # No deforestation, just seasonal atmospheric / illumination slight jitter
        t2_rgb = np.clip(t1_rgb * np.random.uniform(0.96, 1.04) + np.random.normal(0, 0.01, t1_rgb.shape), 0, 1)
        t2_nir = np.clip(t1_nir * np.random.uniform(0.96, 1.04) + np.random.normal(0, 0.01, t1_nir.shape), 0, 1)

    # Scale to 8-bit RGB representation
    t1_rgb_uint8 = (np.clip(t1_rgb, 0, 1) * 255).astype(np.uint8)
    t2_rgb_uint8 = (np.clip(t2_rgb, 0, 1) * 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Pack NIR into 4th channel (RGBA or saved separately)
    t1_rgba = np.dstack([t1_rgb_uint8, (np.clip(t1_nir, 0, 1) * 255).astype(np.uint8)])
    t2_rgba = np.dstack([t2_rgb_uint8, (np.clip(t2_nir, 0, 1) * 255).astype(np.uint8)])

    return t1_rgba, t2_rgba, mask_uint8

def build_full_dataset(train_count: int = 200, val_count: int = 40, test_count: int = 40):
    scene_types = ["fishbone", "clearcut", "selective_logging", "wildfire", "intact_forest"]
    
    splits = {
        "train": (TRAIN_DIR, train_count),
        "val": (VAL_DIR, val_count),
        "test": (TEST_DIR, test_count)
    }

    metadata = []
    
    for split_name, (target_dir, count) in splits.items():
        print(f"Generating {count} samples for split: {split_name}...")
        for i in range(count):
            st = np.random.choice(scene_types, p=[0.3, 0.3, 0.15, 0.15, 0.1])
            t1, t2, mask = generate_satellite_scene(scene_type=st, img_size=256)
            
            sample_id = f"patch_{split_name}_{i+1:04d}_{st}"
            t1_path = os.path.join(target_dir, "t1", f"{sample_id}_t1.png")
            t2_path = os.path.join(target_dir, "t2", f"{sample_id}_t2.png")
            mask_path = os.path.join(target_dir, "masks", f"{sample_id}_mask.png")

            Image.fromarray(t1).save(t1_path)
            Image.fromarray(t2).save(t2_path)
            Image.fromarray(mask).save(mask_path)

            deforested_pixels = int(np.sum(mask > 0))
            loss_percent = round((deforested_pixels / (256 * 256)) * 100, 2)

            metadata.append({
                "sample_id": sample_id,
                "split": split_name,
                "scene_type": st,
                "loss_percentage": loss_percent,
                "t1_file": os.path.relpath(t1_path, DATASET_DIR),
                "t2_file": os.path.relpath(t2_path, DATASET_DIR),
                "mask_file": os.path.relpath(mask_path, DATASET_DIR)
            })

    # Also generate prominent benchmark sample pairs for GUI showcase
    curated_presets = [
        ("01_amazon_fishbone", "fishbone", "Amazon Basin (Rondônia, Brazil) - Road Corridor Logging"),
        ("02_borneo_clearcut", "clearcut", "Borneo Rainforest - Palm Oil Agricultural Clear-cutting"),
        ("03_congo_selective", "selective_logging", "Congo Basin - Selective Commercial Timber Harvesting"),
        ("04_cerrado_fire", "wildfire", "Cerrado Savannah - Anthropogenic Wildfire & Burn Scar"),
        ("05_western_ghats_intact", "intact_forest", "Western Ghats, India - Protected Intact Rainforest Reserve")
    ]

    preset_meta = []
    for p_id, st, desc in curated_presets:
        t1, t2, mask = generate_satellite_scene(scene_type=st, img_size=512)
        t1_path = os.path.join(SAMPLES_DIR, "t1", f"{p_id}_t1.png")
        t2_path = os.path.join(SAMPLES_DIR, "t2", f"{p_id}_t2.png")
        mask_path = os.path.join(SAMPLES_DIR, "masks", f"{p_id}_mask.png")

        Image.fromarray(t1).save(t1_path)
        Image.fromarray(t2).save(t2_path)
        Image.fromarray(mask).save(mask_path)

        deforested_pixels = int(np.sum(mask > 0))
        loss_percent = round((deforested_pixels / (512 * 512)) * 100, 2)
        # Approximate hectares (assuming Sentinel-2 10m pixel: 1 pixel = 100 m² = 0.01 ha)
        hectares_lost = round(deforested_pixels * 0.01, 2)

        preset_meta.append({
            "id": p_id,
            "scene_name": desc,
            "scene_type": st,
            "loss_percentage": loss_percent,
            "hectares_lost": hectares_lost,
            "t1_file": f"sample_pairs/t1/{p_id}_t1.png",
            "t2_file": f"sample_pairs/t2/{p_id}_t2.png",
            "mask_file": f"sample_pairs/masks/{p_id}_mask.png"
        })

    with open(os.path.join(DATASET_DIR, "dataset_manifest.json"), "w") as f:
        json.dump({"samples": metadata, "curated_presets": preset_meta}, f, indent=2)

    print("Successfully generated all dataset splits and curated presets!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Sentinel-2 satellite dataset")
    parser.add_argument("--train-count", type=int, default=200, help="Number of training scene pairs (default: 200)")
    parser.add_argument("--val-count", type=int, default=40, help="Number of validation scene pairs (default: 40)")
    parser.add_argument("--test-count", type=int, default=40, help="Number of test scene pairs (default: 40)")
    args = parser.parse_args()
    build_full_dataset(train_count=args.train_count, val_count=args.val_count, test_count=args.test_count)
