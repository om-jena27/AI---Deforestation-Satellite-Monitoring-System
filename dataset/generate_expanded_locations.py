"""
dataset/generate_expanded_locations.py
======================================
Generates high-resolution 512x512 bi-temporal satellite scenes for 13 additional
worldwide deforestation hotspots (locations 06 to 18), completing an 18-location
global deforestation monitoring catalog.
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset.generate_dataset import generate_satellite_scene

SAMPLES_DIR = ROOT_DIR / "dataset" / "sample_pairs"

NEW_LOCATIONS = [
    ("06_amazon_para_pasture", "clearcut", "Amazon Basin (Pará, Brazil) - Cattle Ranching & Pasture Conversion", -3.25, -52.21, 10.0),
    ("07_mato_grosso_soy", "clearcut", "Mato Grosso (Brazil) - Industrial Soy & Grain Agricultural Frontier", -12.80, -55.80, 10.0),
    ("08_sumatra_peatland", "fishbone", "Sumatra (Riau, Indonesia) - Peatland Drainage & Acacia Concessions", 0.50, 101.50, 10.0),
    ("09_congo_slash_burn", "selective_logging", "Congo Basin (Mai-Ndombe, DRC) - Shifting Cultivation & Charcoal Logging", -2.15, 18.25, 10.0),
    ("10_cameroon_timber_road", "fishbone", "Congo Basin (Cameroon) - Selective Commercial Timber Extraction", 3.10, 12.80, 10.0),
    ("11_gran_chaco_dry_forest", "clearcut", "Gran Chaco (Paraguay & Argentina) - Dry Forest Clearing for Cattle", -22.50, -60.80, 10.0),
    ("12_madagascar_atsinanana", "selective_logging", "Madagascar (Atsinanana) - Rosewood & Slash-and-Burn Tavy", -18.40, 48.50, 10.0),
    ("13_sundarbans_mangrove", "clearcut", "Sundarbans Delta (India / Bangladesh) - Mangrove Degradation & Salinity", 21.90, 88.90, 10.0),
    ("14_cambodia_prey_lang", "selective_logging", "Prey Lang Forest (Cambodia) - Illegal Luxury Timber Poaching", 13.30, 105.30, 10.0),
    ("15_guatemala_maya_biosphere", "clearcut", "Maya Biosphere Reserve (Guatemala) - Transboundary Cattle Encroachment", 17.25, -90.15, 10.0),
    ("16_alberta_boreal_tar_sands", "fishbone", "Boreal Forest (Alberta, Canada) - Tar Sands & Linear Seismic Clearings", 57.00, -111.50, 10.0),
    ("17_siberian_taiga_wildfire", "wildfire", "Siberian Taiga (Krasnoyarsk, Russia) - Megafire Burn Scar & Permafrost Melt", 60.50, 95.20, 10.0),
    ("18_australian_bushfire", "wildfire", "Australian Bush (New South Wales) - Megabushfire Canopy Disturbance", -33.50, 150.30, 10.0),
]

def generate_all_locations():
    print("=" * 65)
    print("[GEN] Generating Satellite Scenes for Worldwide Hotspot Locations...")
    print("=" * 65)

    np.random.seed(1337)
    for p_id, st_type, desc, lat, lon, res in NEW_LOCATIONS:
        t1_path = SAMPLES_DIR / "t1" / f"{p_id}_t1.png"
        t2_path = SAMPLES_DIR / "t2" / f"{p_id}_t2.png"
        m_path = SAMPLES_DIR / "masks" / f"{p_id}_mask.png"

        t1, t2, mask = generate_satellite_scene(scene_type=st_type, img_size=512)
        Image.fromarray(t1).save(t1_path)
        Image.fromarray(t2).save(t2_path)
        Image.fromarray(mask).save(m_path)

        loss_px = int(np.sum(mask > 0))
        loss_pct = round((loss_px / (512 * 512)) * 100, 2)
        loss_ha = round(loss_px * (res * res) / 10000.0, 2)
        print(f"   [OK] Generated {p_id}: {loss_px:,} px loss ({loss_pct:.2f}% | {loss_ha:,.2f} ha) | Lat: {lat:.2f}, Lon: {lon:.2f}")

    print("=" * 65)
    print(f"[OK] All {len(NEW_LOCATIONS)} new global hotspot locations generated successfully!")
    print("=" * 65)

if __name__ == "__main__":
    generate_all_locations()
