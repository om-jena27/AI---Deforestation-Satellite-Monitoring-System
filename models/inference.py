"""
models/inference.py
===================
Deep Learning Model Management & Inference Pipeline.

Handles:
- Checkpoint auto-discovery and validation
- Clean distinction between REAL MODEL INFERENCE and DEMO MODE
- GPU acceleration with automatic CPU fallback
- torch.no_grad() inference caching
- Dynamic demo mode override for UI validation
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from .unet import ForestUNet

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHECKPOINT_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "checkpoints", "unet_deforestation.pth"),
    os.path.join(BASE_DIR, "models", "unet_deforestation.pth"),
    os.path.join(BASE_DIR, "weights", "unet_forest.pt"),
    os.path.join(BASE_DIR, "weights", "unet_deforestation.pth"),
]


class ModelManager:
    """
    Manages deep learning model lifecycle, checkpoint detection, and inference execution.
    Strictly distinguishes between REAL MODEL INFERENCE mode and DEMO MODE.
    """
    def __init__(self, force_demo: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ForestUNet | None = None
        self.is_trained = False
        self.checkpoint_path: str | None = None
        self.force_demo = force_demo
        self.mode_label = "DEMO / MODEL NOT TRAINED"
        self._initialize()

    def _initialize(self) -> None:
        """Instantiate ForestUNet and load weights if a verified checkpoint is present."""
        self.model = ForestUNet(in_channels=3, out_channels=1)

        if not self.force_demo:
            for path in CHECKPOINT_CANDIDATES:
                if os.path.exists(path) and os.path.getsize(path) > 10_000:
                    try:
                        state_dict = torch.load(path, map_location=self.device)
                        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                            state_dict = state_dict["model_state_dict"]

                        # Validate parameter compatibility
                        model_dict = self.model.state_dict()
                        matching = {
                            k: v for k, v in state_dict.items()
                            if k in model_dict and v.shape == model_dict[k].shape
                        }

                        # Require at least 90% parameter match for certified trained status
                        if len(matching) >= len(model_dict) * 0.9:
                            model_dict.update(matching)
                            self.model.load_state_dict(model_dict)
                            self.checkpoint_path = path
                            self.is_trained = True
                            self.mode_label = f"REAL MODEL INFERENCE ({os.path.basename(path)})"
                            break
                    except Exception as exc:
                        print(f"[ModelManager] Failed loading {path}: {exc}")

        if not self.is_trained:
            if self.force_demo:
                self.mode_label = "DEMO / MODEL NOT TRAINED (Manual Toggle)"
            else:
                self.mode_label = "DEMO / MODEL NOT TRAINED (No Valid Checkpoint)"

        self.model.to(self.device).eval()

    def set_force_demo(self, force: bool) -> None:
        """Allows dynamic toggling between real inference and demo mode for UI testing."""
        if self.force_demo != force:
            self.force_demo = force
            self._initialize()

    def predict_forest_probability(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Runs real deep learning inference using torch.no_grad().
        
        Args:
            img_tensor: Tensor of shape (1, 3, H, W) normalized to [0.0, 1.0].
            
        Returns:
            np.ndarray: 2D probability map (H, W) in range [0.0, 1.0].
        """
        if not self.is_trained:
            raise RuntimeError("Cannot execute real model inference in DEMO mode.")

        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            out = self.model(img_tensor)
            prob_map = out.squeeze().cpu().numpy()

        return np.clip(prob_map, 0.0, 1.0).astype(np.float32)

    def get_summary(self) -> dict[str, Any]:
        is_synth = "unet_deforestation.pth" in (self.checkpoint_path or "")
        return {
            "is_trained": self.is_trained,
            "mode_label": self.mode_label,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "architecture": "U-Net (ForestUNet, 4-stage encoder-decoder)",
            "input_channels": 3,
            "output_channels": 1,
            "training_domain": "Synthetic Procedural Benchmark (dataset/train)" if is_synth else "Real-World Satellite Data",
            "real_world_validated": not is_synth,
        }

    def check_domain_compatibility(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Validates whether the active checkpoint was trained on data comparable to the current input.
        Warns the user when a synthetically-trained model is applied to real-world satellite imagery.
        """
        metadata = metadata or {}
        is_synthetic_input = metadata.get("is_synthetic", False)
        checkpoint_is_synthetic = "unet_deforestation.pth" in (self.checkpoint_path or "")
        has_domain_mismatch = (not is_synthetic_input) and checkpoint_is_synthetic and self.is_trained

        warning = None
        if has_domain_mismatch:
            warning = (
                "⚠️ Model Domain Mismatch Warning: The active U-Net checkpoint ('unet_deforestation.pth') "
                "was trained exclusively on procedurally simulated benchmark imagery (dataset/train). "
                "Real satellite imagery contains atmospheric scattering, cloud shadows, and seasonal phenology "
                "that can affect generalization. Retraining with labelled real-world GeoTIFF data using "
                "`python -m training.train_real` is recommended for operational monitoring."
            )

        return {
            "checkpoint_trained": self.is_trained,
            "checkpoint_name": os.path.basename(self.checkpoint_path) if self.checkpoint_path else "None",
            "training_domain": "Synthetic Procedural Benchmark" if checkpoint_is_synthetic else "Real-World Satellite Imagery",
            "input_domain": "Synthetic Benchmark" if is_synthetic_input else "Real-World Satellite Data",
            "has_domain_mismatch": has_domain_mismatch,
            "warning": warning,
            "recommended_workflow": "training/train_real.py",
        }


# Global singleton cache
_GLOBAL_MANAGER: ModelManager | None = None


def get_model_manager(force_demo: bool = False) -> ModelManager:
    global _GLOBAL_MANAGER
    if (
        _GLOBAL_MANAGER is None
        or _GLOBAL_MANAGER.force_demo != force_demo
        or not hasattr(_GLOBAL_MANAGER, "check_domain_compatibility")
    ):
        _GLOBAL_MANAGER = ModelManager(force_demo=force_demo)
    return _GLOBAL_MANAGER
