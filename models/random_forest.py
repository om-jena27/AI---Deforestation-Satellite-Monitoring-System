import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, jaccard_score
from .spectral_indices import compute_spectral_features

class RandomForestDeforestationDetector:
    """
    Random Forest Deforestation Classifier based on multispectral vegetation indices.
    Inspired by Arko et al. (Nature Scientific Reports 2026).
    """
    def __init__(self, n_estimators=80, max_depth=14, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state
        )
        self.is_trained = False

    def extract_features(self, t1_arr, t2_arr):
        """
        Builds feature matrix X of shape (H*W, N_FEATURES)
        """
        feats = compute_spectral_features(t1_arr, t2_arr)
        h, w = t1_arr.shape[:2]

        t1_r = t1_arr[:, :, 0].flatten() / 255.0
        t1_g = t1_arr[:, :, 1].flatten() / 255.0
        t1_b = t1_arr[:, :, 2].flatten() / 255.0
        t1_nir = (t1_arr[:, :, 3].flatten() / 255.0) if t1_arr.shape[2] > 3 else t1_g * 1.5

        t2_r = t2_arr[:, :, 0].flatten() / 255.0
        t2_g = t2_arr[:, :, 1].flatten() / 255.0
        t2_b = t2_arr[:, :, 2].flatten() / 255.0
        t2_nir = (t2_arr[:, :, 3].flatten() / 255.0) if t2_arr.shape[2] > 3 else t2_g * 1.5

        ndvi1 = feats["ndvi_t1"].flatten()
        ndvi2 = feats["ndvi_t2"].flatten()
        d_ndvi = feats["delta_ndvi"].flatten()
        savi1 = feats["savi_t1"].flatten()
        savi2 = feats["savi_t2"].flatten()
        d_savi = feats["delta_savi"].flatten()
        d_rgb = feats["diff_rgb"].flatten()
        d_red = feats["diff_red"].flatten()

        X = np.column_stack([
            t1_r, t1_g, t1_b, t1_nir,
            t2_r, t2_g, t2_b, t2_nir,
            ndvi1, ndvi2, d_ndvi,
            savi1, savi2, d_savi,
            d_rgb, d_red
        ])
        return X

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict_mask(self, t1_arr, t2_arr, threshold=0.5):
        h, w = t1_arr.shape[:2]
        X = self.extract_features(t1_arr, t2_arr)
        proba = self.model.predict_proba(X)[:, 1]
        mask = (proba >= threshold).astype(np.uint8).reshape(h, w)
        prob_map = proba.reshape(h, w)
        return mask, prob_map

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
