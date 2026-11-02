# 🛰️ Deforestation Detection & Canopy Segmentation Dataset

Welcome to the unified satellite dataset directory for the **AI-Based Deforestation Detection & Environmental Monitoring System**.

This folder houses all training, validation, testing, and curated benchmark data utilized by the machine learning models (U-Net Deep Learning & Random Forest).

---

## 📁 Directory Structure & Overview

```text
dataset/
├── Forest-Change-dataset.zip           # Complete offline archive of real satellite dataset (23.5 MB)
├── dataset_manifest.json               # Master JSON index (18 presets, 334 real pairs, 329 synthetic pairs)
├── build_manifest.py                   # Automated script to inspect and rebuild the manifest index
├── generate_dataset.py                 # Multi-spectral (4-band RGB+NIR) procedural dataset generator
├── generate_expanded_locations.py      # Generator for the 18 global deforestation hotspot case studies
├── download_online_dataset.py          # Automated Hugging Face downloader & unpacker
│
├── sample_pairs/                       # 18 Curated Global Hotspot Case Studies (512x512 PNG, 10m GSD)
│   ├── t1/                             # Pre-disturbance baseline satellite scenes (T1)
│   ├── t2/                             # Post-disturbance satellite scenes (T2)
│   └── masks/                          # Ground-truth binary deforestation disturbance masks
│
├── online_forest_change/               # Authentic Real Satellite Dataset (JimmyBrocko/Forest-Change)
│   └── Forest-Change-dataset/
│       └── images/
│           ├── train/                  # 270 bi-temporal pairs (A: T1, B: T2, label: Ground Truth)
│           ├── val/                    # 31 bi-temporal pairs (A: T1, B: T2, label: Ground Truth)
│           └── test/                   # 33 bi-temporal pairs (A: T1, B: T2, label: Ground Truth)
│
├── real_satellite_samples/             # Real Georeferenced GeoTIFF & Rapid Showcase Benchmarks
│   ├── t1/                             # Sentinel-2 L2A BOA Reflectance GeoTIFF (EPSG:32720)
│   ├── t2/                             # Sentinel-2 L2A BOA Reflectance GeoTIFF (EPSG:32720)
│   ├── masks/                          # Deforestation Ground Truth GeoTIFF
│   └── online_scenes/                  # 5 Showcase real satellite scenes for instant web demo testing
│
├── train/                              # Multi-spectral Sentinel-2 Training Split (200 pairs, 256x256 RGBA)
│   ├── t1/                             # Pre-clearing 4-band imagery (Red, Green, Blue, Near-Infrared)
│   ├── t2/                             # Post-clearing 4-band imagery
│   └── masks/                          # Pixel-level deforestation disturbance ground truth
│
├── val/                                # Multi-spectral Validation Split (64 pairs, 256x256 RGBA)
│   ├── t1/, t2/, masks/
│
└── test/                               # Multi-spectral Independent Test Split (16 pairs, 256x256 RGBA)
    ├── t1/, t2/, masks/
```

---

## 🌍 18 Curated Global Hotspot Locations

The dataset features 18 curated real-world frontline deforestation case studies spanning 6 continents:

| ID | Location | Coordinates | Deforestation Type | Loss % | Area Lost |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **01** | Amazon Basin (Rondônia, Brazil) | `10.87° S, 62.83° W` | Fishbone Road Logging | 5.92% | 155.16 ha |
| **02** | Borneo Rainforest (Kalimantan, Indonesia) | `0.13° N, 114.28° E` | Palm Oil Clear-Cutting | 8.06% | 211.40 ha |
| **03** | Congo Basin (DR Congo) | `0.75° N, 24.50° E` | Selective Timber Extraction | 3.36% | 88.16 ha |
| **04** | Cerrado Savannah (Brazil) | `12.50° S, 47.00° W` | Agricultural Wildfire & Burn Scar | 5.33% | 139.78 ha |
| **05** | Western Ghats (Karnataka, India) | `13.15° N, 75.25° E` | Undisturbed Intact Canopy (Control) | 0.00% | 0.00 ha |
| **06** | Amazon Basin (Pará, Brazil) | `3.25° S, 52.21° W` | Cattle Ranching & Pasture Conversion | 6.88% | 180.41 ha |
| **07** | Mato Grosso Frontier (Brazil) | `12.65° S, 55.85° W` | Industrial Soy Monoculture Expansion | 5.21% | 136.68 ha |
| **08** | Sumatra Peat Swamp (Riau, Indonesia) | `0.58° N, 101.45° E` | Canal Drainage & Pulpwood Clearing | 6.62% | 173.44 ha |
| **09** | Mai-Ndombe (DR Congo) | `1.95° S, 18.27° E` | Shifting Slash-and-Burn Mosaic | 4.83% | 126.66 ha |
| **10** | South Cameroon Forest Concessions | `2.92° N, 11.51° E` | Industrial Concession Arterial Roads | 9.13% | 239.24 ha |
| **11** | Gran Chaco Dry Forest (Argentina/Paraguay) | `24.89° S, 61.54° W` | Dry Subtropical Pasture Clearing | 8.86% | 232.14 ha |
| **12** | Rainforests of the Atsinanana (Madagascar) | `18.78° S, 48.45° E` | Montane Tavy Slash-and-Burn | 3.40% | 89.22 ha |
| **13** | Sundarbans Biosphere (India/Bangladesh) | `21.95° N, 88.85° E` | Mangrove Recession & Aquaculture | 7.77% | 203.56 ha |
| **14** | Prey Lang Sanctuary (Cambodia) | `13.12° N, 105.48° E` | Illegal Logging & Rubber Encroachment | 3.58% | 93.88 ha |
| **15** | Maya Biosphere Reserve (Petén, Guatemala) | `17.25° N, 90.15° W` | Frontier Narco-Ranching & Airfields | 5.66% | 148.28 ha |
| **16** | Athabasca Boreal (Alberta, Canada) | `57.02° N, 111.49° W` | Open-Pit Tar Sands & Seismic Lines | 8.08% | 211.80 ha |
| **17** | Siberian Taiga (Sakha Republic, Russia) | `62.15° N, 129.74° E` | Climate-Amplified Megafire Perimeter | 5.30% | 139.01 ha |
| **18** | Gippsland & Blue Mtns (Australia) | `37.25° S, 148.55° E` | Extreme Crown-Fire Burn Scar | 5.33% | 139.62 ha |

---

## 🛰️ Sensor & Multi-Spectral Specifications

1. **Resolution & Geometry:**
   - Spatial Resolution: 10.0 meters Ground Sample Distance (GSD).
   - Coordinate Reference Systems: EPSG:4326 (WGS 84 geographic coordinates) and EPSG:32720 (UTM Zone 20S projected planar coordinates).
   - Bounding Box Dimensions: $512 \times 512$ pixels covers $5,120 \times 5,120\text{ m} = 2,621.44\text{ hectares} = 26.21\text{ km}^2$.

2. **Spectral Bands:**
   - **Band 2 (Blue):** $490\text{ nm}$ — atmospheric Rayleigh scattering & water distinction.
   - **Band 3 (Green):** $560\text{ nm}$ — healthy vegetation chlorophyll peak.
   - **Band 4 (Red):** $665\text{ nm}$ — chlorophyll absorption valley.
   - **Band 8 (Near-Infrared / NIR):** $842\text{ nm}$ — high mesophyll cellular reflectance.
   - **Vegetation Indices:** $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$ and $\text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$.

---

## ⚡ How to Train or Retrain Models

The training pipeline in the project root (`train.py`) automatically ingests both the offline multi-spectral data and authentic satellite imagery:

```bash
# Train ForestUNet on all available datasets:
python train.py --epochs 6 --batch-size 8 --lr 0.001

# Inspect the dataset manifest:
python dataset/build_manifest.py
```

Checkpoints will be written to:
- `models/checkpoints/unet_deforestation.pth`
- `weights/unet_best.pt`
