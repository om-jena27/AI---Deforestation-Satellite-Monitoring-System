import os
import glob
import numpy as np
from PIL import Image
from models.random_forest import RandomForestDeforestationDetector

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def train_rf_fast():
    print("Fast training Random Forest classifier...")
    rf = RandomForestDeforestationDetector(n_estimators=50, max_depth=12)
    
    t1_files = sorted(glob.glob(os.path.join(DATASET_DIR, "train", "t1", "*.png")))[:40]
    t2_files = sorted(glob.glob(os.path.join(DATASET_DIR, "train", "t2", "*.png")))[:40]
    mask_files = sorted(glob.glob(os.path.join(DATASET_DIR, "train", "masks", "*.png")))[:40]
    
    X_list, y_list = [], []
    for f1, f2, fm in zip(t1_files, t2_files, mask_files):
        t1 = np.array(Image.open(f1))
        t2 = np.array(Image.open(f2))
        m = (np.array(Image.open(fm)) > 127).astype(np.uint8)
        if m.ndim == 3:
            m = m[:, :, 0]
            
        X = rf.extract_features(t1, t2)
        y = m.flatten()
        
        idx = np.random.choice(len(y), size=int(len(y) * 0.05), replace=False)
        X_list.append(X[idx])
        y_list.append(y[idx])
        
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    print(f"Fitting on {len(X_train)} samples...")
    rf.fit(X_train, y_train)
    
    out_path = os.path.join(WEIGHTS_DIR, "random_forest.joblib")
    rf.save(out_path)
    print(f"Saved Random Forest to {out_path}!")

if __name__ == "__main__":
    train_rf_fast()
