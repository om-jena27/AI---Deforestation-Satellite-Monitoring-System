import os
import glob
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.unet import DeforestationUNet, CombinedBCEDiceLoss
from models.siamese_cd import SiameseChangeNet
from models.random_forest import RandomForestDeforestationDetector
from models.spectral_indices import baseline_ndvi_threshold_detector, compute_spectral_features
from models.evaluate import calculate_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

class SatelliteChangeDataset(Dataset):
    def __init__(self, split_dir, img_size=256):
        self.t1_files = sorted(glob.glob(os.path.join(split_dir, "t1", "*.png")))
        self.t2_files = sorted(glob.glob(os.path.join(split_dir, "t2", "*.png")))
        self.mask_files = sorted(glob.glob(os.path.join(split_dir, "masks", "*.png")))
        self.img_size = img_size
        assert len(self.t1_files) == len(self.t2_files) == len(self.mask_files), "Mismatch in dataset files!"

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        t1 = np.array(Image.open(self.t1_files[idx]), dtype=np.float32) / 255.0
        t2 = np.array(Image.open(self.t2_files[idx]), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.mask_files[idx]), dtype=np.float32) / 255.0

        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        elif mask.ndim == 3 and mask.shape[2] > 1:
            mask = mask[:, :, :1]

        # PyTorch format: (C, H, W)
        t1_tensor = torch.from_numpy(t1).permute(2, 0, 1).float()
        t2_tensor = torch.from_numpy(t2).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()

        # Concatenated bi-temporal tensor for standard U-Net
        concat_t1_t2 = torch.cat([t1_tensor, t2_tensor], dim=0)

        return {
            "t1": t1_tensor,
            "t2": t2_tensor,
            "concat": concat_t1_t2,
            "mask": mask_tensor
        }

def train_unet(train_loader, val_loader, epochs=8, device="cpu"):
    print("\n--- Training Deep Learning U-Net (Bi-Temporal) ---")
    model = DeforestationUNet(in_channels=8, out_channels=1).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_path = os.path.join(WEIGHTS_DIR, "unet_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["concat"].to(device)
            y = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["concat"].to(device)
                y = batch["mask"].to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"U-Net Training Finished! Best weights saved to: {best_path}")
    return model

def train_siamese(train_loader, val_loader, epochs=8, device="cpu"):
    print("\n--- Training Siamese Change Detection Network (Siam-Diff) ---")
    model = SiameseChangeNet(in_channels=4, out_channels=1).to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_path = os.path.join(WEIGHTS_DIR, "siamese_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            y = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(t1, t2)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device)
                t2 = batch["t2"].to(device)
                y = batch["mask"].to(device)
                preds = model(t1, t2)
                loss = criterion(preds, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"Siamese Net Training Finished! Best weights saved to: {best_path}")
    return model

def train_random_forest(train_dataset, sample_fraction=0.03):
    print("\n--- Training Random Forest Pixel & Spectral Index Classifier ---")
    rf = RandomForestDeforestationDetector(n_estimators=75, max_depth=12)
    X_list, y_list = [], []

    # Sample sub-pixels from training set for rapid, robust training
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        # Convert back to uint8 HWC
        t1_np = (item["t1"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        t2_np = (item["t2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = (item["mask"].squeeze(0).numpy() > 0.5).astype(np.uint8)

        X_full = rf.extract_features(t1_np, t2_np)
        y_full = mask_np.flatten()

        # Subsample to keep training fast
        n_pts = len(y_full)
        indices = np.random.choice(n_pts, size=int(n_pts * sample_fraction), replace=False)
        X_list.append(X_full[indices])
        y_list.append(y_full[indices])

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    print(f"Training Random Forest on {len(X_train)} spectral feature samples...")
    rf.fit(X_train, y_train)

    rf_path = os.path.join(WEIGHTS_DIR, "random_forest.joblib")
    rf.save(rf_path)
    print(f"Random Forest Saved to: {rf_path}")
    return rf

def benchmark_models(test_dataset, unet_model, siamese_model, rf_model, device="cpu"):
    print("\n================ BENCHMARKING ON TEST SET ================")
    results = {
        "UNet": [],
        "SiameseChangeNet": [],
        "RandomForest": [],
        "BaselineNDVI": []
    }

    unet_model.eval()
    siamese_model.eval()

    for i in range(len(test_dataset)):
        item = test_dataset[i]
        t1 = item["t1"].unsqueeze(0).to(device)
        t2 = item["t2"].unsqueeze(0).to(device)
        concat = item["concat"].unsqueeze(0).to(device)
        gt_mask = item["mask"].squeeze().numpy() > 0.5

        t1_np = (item["t1"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        t2_np = (item["t2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # 1. U-Net
        with torch.no_grad():
            pred_unet = (unet_model(concat).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
        m_unet = calculate_metrics(gt_mask, pred_unet)
        results["UNet"].append(m_unet)

        # 2. Siamese Net
        with torch.no_grad():
            pred_siam = (siamese_model(t1, t2).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
        m_siam = calculate_metrics(gt_mask, pred_siam)
        results["SiameseChangeNet"].append(m_siam)

        # 3. Random Forest
        pred_rf, _ = rf_model.predict_mask(t1_np, t2_np)
        m_rf = calculate_metrics(gt_mask, pred_rf)
        results["RandomForest"].append(m_rf)

        # 4. Baseline NDVI Threshold
        feats = compute_spectral_features(t1_np, t2_np)
        pred_ndvi = baseline_ndvi_threshold_detector(feats["delta_ndvi"], threshold=0.25)
        m_ndvi = calculate_metrics(gt_mask, pred_ndvi)
        results["BaselineNDVI"].append(m_ndvi)

    # Average metrics
    summary = {}
    for model_name, m_list in results.items():
        summary[model_name] = {
            "Accuracy": round(float(np.mean([m["accuracy"] for m in m_list])), 4),
            "Precision": round(float(np.mean([m["precision"] for m in m_list])), 4),
            "Recall": round(float(np.mean([m["recall"] for m in m_list])), 4),
            "F1_Dice": round(float(np.mean([m["f1_dice"] for m in m_list])), 4),
            "IoU": round(float(np.mean([m["iou"] for m in m_list])), 4)
        }

    print("\n--- FINAL BENCHMARK SUMMARY TABLE ---")
    print(f"{'Model Architecture':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 / Dice':<10} | {'IoU':<10}")
    print("-" * 80)
    for name, s in summary.items():
        print(f"{name:<22} | {s['Accuracy']:<10} | {s['Precision']:<10} | {s['Recall']:<10} | {s['F1_Dice']:<10} | {s['IoU']:<10}")

    with open(os.path.join(WEIGHTS_DIR, "benchmark_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    train_ds = SatelliteChangeDataset(os.path.join(DATASET_DIR, "train"))
    val_ds = SatelliteChangeDataset(os.path.join(DATASET_DIR, "val"))
    test_ds = SatelliteChangeDataset(os.path.join(DATASET_DIR, "test"))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    unet = train_unet(train_loader, val_loader, epochs=6, device=device)
    siamese = train_siamese(train_loader, val_loader, epochs=6, device=device)
    rf = train_random_forest(train_ds)

    benchmark_models(test_ds, unet, siamese, rf, device=device)

if __name__ == "__main__":
    main()
