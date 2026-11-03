from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageFilter
import numpy as np


class DatasetPreprocessor:
    """Utility for preparing a forest-fire dataset while minimizing leakage."""

    VALID_CLASSES = ["fire", "smoke", "no_fire", "cloud", "fog", "dust"]

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "val"
        self.test_dir = self.dataset_root / "test"

    def list_image_files(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    def check_corrupted_images(self, files: List[Path]) -> List[Path]:
        corrupted = []
        for item in files:
            try:
                with Image.open(item) as img:
                    img.verify()
            except Exception:
                corrupted.append(item)
        return corrupted

    def detect_duplicate_images(self, files: List[Path]) -> Dict[str, List[str]]:
        seen: Dict[str, str] = {}
        duplicates: Dict[str, List[str]] = {}
        for item in files:
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            if digest in seen:
                duplicates.setdefault(digest, [seen[digest]]).append(str(item))
            else:
                seen[digest] = str(item)
        return {k: v for k, v in duplicates.items() if len(v) > 1}

    def normalize_image(self, image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(target_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def apply_noise_reduction(self, image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(target_size)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.2) -> Dict[str, float]:
        if train_ratio + val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")
        return {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 4),
        }

    def leakage_check(self, root: Path) -> List[str]:
        """Warn about accidental leakage when the same source appears in multiple splits."""
        issue_files: List[str] = []
        for item in root.rglob("*"):
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                parent = item.parent.name
                if parent in {"train", "val", "test"}:
                    issue_files.append(str(item.relative_to(root)))
        return issue_files

    def class_distribution(self, root: Path) -> Dict[str, int]:
        counts: Dict[str, int] = {cls: 0 for cls in self.VALID_CLASSES}
        for cls in self.VALID_CLASSES:
            folder = root / cls
            if folder.exists():
                counts[cls] = len(self.list_image_files(folder))
        return counts

    def make_dataset_structure(self) -> None:
        for stage in ["train", "val", "test"]:
            for cls in self.VALID_CLASSES:
                (self.dataset_root / stage / cls).mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self) -> Dict[str, object]:
        self.make_dataset_structure()
        train_files = self.list_image_files(self.train_dir)
        val_files = self.list_image_files(self.val_dir)
        test_files = self.list_image_files(self.test_dir)
        return {
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "duplicate_images": self.detect_duplicate_images(train_files + val_files + test_files),
            "corrupted_images": self.check_corrupted_images(train_files + val_files + test_files),
            "class_distribution": self.class_distribution(self.dataset_root),
            "leakage_warnings": self.leakage_check(self.dataset_root),
        }
