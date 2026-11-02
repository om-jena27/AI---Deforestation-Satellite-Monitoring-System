"""
dataset/download_online_dataset.py
==================================
Automated Downloader & Preprocessor for Open-Access Satellite Deforestation Datasets:
1. Downloads 'Forest-Change' benchmark dataset from Hugging Face:
   - 334 real-world Earth Observation bi-temporal satellite image pairs (480x480)
   - Pre-event (A), Post-event (B), and Pixel-level Deforestation Ground Truth (label)
   - Real tropical forest monitoring data from Google Earth Engine & Sentinel satellites
2. Unpacks and organizes into dataset/online_forest_change/
3. Extracts real satellite showcase scenes into dataset/real_satellite_samples/online_scenes/
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
ONLINE_DIR = DATASET_DIR / "online_forest_change"
DOWNLOADS_DIR = DATASET_DIR / "downloads"
REAL_SHOWCASE_DIR = DATASET_DIR / "real_satellite_samples" / "online_scenes"

HUGGINGFACE_DATASET_URL = (
    "https://huggingface.co/datasets/JimmyBrocko/Forest-Change/resolve/main/Forest-Change-dataset.zip"
)

def download_and_setup_dataset(force: bool = False):
    print("=" * 70)
    print("🛰️  ONLINE SATELLITE DEFORESTATION DATASET DOWNLOADER")
    print("    Source: Hugging Face (JimmyBrocko/Forest-Change)")
    print("    Sensors: Sentinel-2 & Google Earth Engine Multi-Temporal Imagery")
    print("=" * 70)

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "Forest-Change-dataset.zip"

    # Step 1: Download if not present
    if not zip_path.exists() or force:
        print(f"\n[1/3] Downloading Forest-Change-dataset.zip (~23.5 MB)...")
        print(f"      URL: {HUGGINGFACE_DATASET_URL}")
        try:
            req = urllib.request.Request(
                HUGGINGFACE_DATASET_URL,
                headers={"User-Agent": "AI-Deforestation-System/2.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_f:
                total_size = int(response.info().get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 512
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"      -> Downloaded {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB ({pct:.1f}%)", end="\r", flush=True)
            print(f"\n      [OK] Download complete: {zip_path.stat().st_size:,} bytes")
        except Exception as exc:
            print(f"\n      [ERROR] Download failed: {exc}")
            print("      Ensure an active internet connection is available.")
            return False
    else:
        print(f"\n[1/3] Archive already present: {zip_path} ({zip_path.stat().st_size:,} bytes)")

    # Step 2: Unpack
    print(f"\n[2/3] Extracting archive contents into {ONLINE_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if not member.startswith("__MACOSX"):
                z.extract(member, ONLINE_DIR)
    print("      [OK] Extraction completed.")

    # Step 3: Populate real showcase scenes for GUI
    print(f"\n[3/3] Preparing showcase scenes in {REAL_SHOWCASE_DIR}...")
    src_test = ONLINE_DIR / "Forest-Change-dataset" / "images" / "test"
    for sub in ["t1", "t2", "masks"]:
        (REAL_SHOWCASE_DIR / sub).mkdir(parents=True, exist_ok=True)

    if src_test.exists():
        test_files = sorted(list((src_test / "A").glob("*.png")))[:5]
        for idx, a_p in enumerate(test_files, 1):
            fname = a_p.name
            b_p = src_test / "B" / fname
            m_p = src_test / "label" / fname
            tag = f"real_sat_scene_{idx:02d}"
            shutil.copy(a_p, REAL_SHOWCASE_DIR / "t1" / f"{tag}_t1.png")
            shutil.copy(b_p, REAL_SHOWCASE_DIR / "t2" / f"{tag}_t2.png")
            shutil.copy(m_p, REAL_SHOWCASE_DIR / "masks" / f"{tag}_mask.png")
            print(f"      -> Preserved showcase scene {idx}: {fname} -> {tag}")

    print("\n" + "=" * 70)
    print("✅ Online satellite dataset successfully integrated into pipeline!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-download even if archive exists")
    args = parser.parse_args()
    download_and_setup_dataset(force=args.force)
