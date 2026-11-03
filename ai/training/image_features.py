from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


IMAGE_SIZE: Tuple[int, int] = (32, 32)
CLASS_NAMES: List[str] = ["fire", "smoke", "no_fire"]


def extract_image_features(image_path: str | Path) -> np.ndarray:
    """Create deterministic RGB features for the scikit-learn image model."""
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize(IMAGE_SIZE)
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0

    channel_means = pixels.mean(axis=(0, 1))
    channel_stds = pixels.std(axis=(0, 1))
    features = np.concatenate([pixels.reshape(-1), channel_means, channel_stds])
    return features.astype(np.float32)
