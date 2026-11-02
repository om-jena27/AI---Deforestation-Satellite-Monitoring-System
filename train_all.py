"""
train_all.py
============
Master Pipeline: Generates expanded satellite dataset and trains AI models:
1. Generates 200 train pairs (400 canopy training images), 40 val pairs, 40 test pairs.
2. Trains Deep Learning ForestUNet (4 epochs, batch size 16, multi-threaded CPU).
3. Trains Random Forest Classifier on >100,000 multispectral pixel signatures.
4. Evaluates all models against the independent test set and updates benchmarks.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def run_command(cmd, desc):
    print("\n" + "=" * 70)
    print(f"[RUN] {desc}")
    print(f"      Command: {' '.join(cmd)}")
    print("=" * 70)
    custom_env = os.environ.copy()
    custom_env["PYTHONIOENCODING"] = "utf-8"
    custom_env["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/Temp"), "matplotlib")
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), env=custom_env)
    if res.returncode != 0:
        print(f"[ERROR] Error during: {desc} (Exit code {res.returncode})")
        sys.exit(res.returncode)

def main():
    print("=" * 70)
    print("AI Deforestation System -- Large-Scale Model Training Pipeline")
    print("Multi-Source: Authentic Online Earth Observation + 18 Global Hotspots")
    print("=" * 70)

    # 1. Download & Unpack Online Satellite Dataset (Hugging Face Forest-Change)
    run_command([sys.executable, "dataset/download_online_dataset.py"],
                "Step 1: Ingesting Real Online Satellite Deforestation Dataset (HuggingFace)")

    # 2. Generate 18 Worldwide Deforestation Hotspots
    run_command([sys.executable, "dataset/generate_expanded_locations.py"],
                "Step 2: Generating 18 Worldwide Deforestation Hotspot Locations")

    # 3. Deep Learning U-Net Training on 940 Satellite Canopy Images
    run_command([sys.executable, "train.py", "--epochs", "3", "--batch-size", "24"], 
                "Step 3: Training PyTorch ForestUNet on 940 Combined Satellite Canopy Images")

    # 4. Random Forest Classifier Training
    run_command([sys.executable, "train_rf.py"], 
                "Step 4: Training Random Forest Classifier on Multispectral Signatures")

    # 5. Benchmark Evaluation
    run_command([sys.executable, "models/run_benchmark.py"], 
                "Step 5: Benchmarking All Evaluated Models on Independent Test Set")

    print("\n" + "=" * 70)
    print("[OK] ALL MODELS TRAINED AND BENCHMARKED ON EXPANDED + ONLINE DATASETS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
