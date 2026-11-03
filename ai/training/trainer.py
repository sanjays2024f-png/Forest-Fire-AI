from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from ai.training.image_features import CLASS_NAMES, extract_image_features


class TrainingEngine:
    """Train the project's real scikit-learn image classifier."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    def _load_split(self, dataset_path: Path, split: str) -> tuple[list, list]:
        paths: list = []
        labels: list = []
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for class_name in CLASS_NAMES:
            for root in (dataset_path / split, dataset_path / split / class_name):
                d = root if root.name == class_name else root / class_name
                if not d.exists():
                    continue
                for path in sorted(d.rglob("*")):
                    if path.is_file() and path.suffix.lower() in image_suffixes and path not in paths:
                        paths.append(path)
                        labels.append(class_name)
                break
        return paths, labels

    def train(self, dataset_dir: str | Path) -> Dict[str, object]:
        import random
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        dataset_path = Path(dataset_dir)
        train_paths, train_labels = self._load_split(dataset_path, "train")
        val_paths, val_labels = self._load_split(dataset_path, "val")
        test_paths, test_labels = self._load_split(dataset_path, "test")
        # fallback: legacy flat layout (dataset/<class> or dataset/images/<class>)
        if not train_paths:
            image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            roots = [dataset_path / "images", dataset_path]
            for class_name in CLASS_NAMES:
                for root in roots:
                    class_dir = root / class_name
                    for path in sorted(class_dir.rglob("*")) if class_dir.exists() else []:
                        if path.is_file() and path.suffix.lower() in image_suffixes and path not in train_paths:
                            train_paths.append(path)
                            train_labels.append(class_name)
        counts = {name: train_labels.count(name) for name in CLASS_NAMES}

        def _fail(reason: str) -> Dict[str, object]:
            return {"device": self.device, "status": "not_trained",
                    "training_completed": False, "reason": reason,
                    "class_counts": counts, "checkpoint_path": None}

        if len(set(train_labels)) != len(CLASS_NAMES):
            return _fail("Expected labeled images under dataset/train/<fire|smoke|no_fire> (all 3 classes).")
        if min(counts.values()) < 2:
            return _fail("At least two images per class are required.")
        if not val_paths or not test_paths:
            return _fail("Validation and test splits required (dataset/val + dataset/test).")

        def _feats(ps: list) -> "np.ndarray":
            return np.vstack([extract_image_features(p) for p in ps])

        train_features = _feats(train_paths)
        val_features = _feats(val_paths)
        test_features = _feats(test_paths)
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.2,
            max_iter=80,
            random_state=42,
        )
        model.fit(train_features, train_labels)
        output_path = dataset_path.parent / "outputs" / "models" / "best_model.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

        def _metrics(yt: list, yp) -> dict:
            return {
                "accuracy": round(float(accuracy_score(yt, yp)), 4),
                "precision": round(float(precision_score(yt, yp, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(yt, yp, average="weighted", zero_division=0)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
                "confusion_matrix": confusion_matrix(yt, yp, labels=CLASS_NAMES).tolist(),
                "confusion_labels": CLASS_NAMES,
            }

        train_pred = model.predict(train_features)
        val_pred = model.predict(val_features)
        test_pred = model.predict(test_features)
        result = {
            "device": self.device,
            "status": "trained",
            "training_completed": True,
            "checkpoint_path": str(output_path),
            "class_names": list(model.classes_),
            "class_order": CLASS_NAMES,
            "sample_count": len(train_paths) + len(val_paths) + len(test_paths),
            "split": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
            "train_metrics": _metrics(train_labels, train_pred),
            "val_metrics": _metrics(val_labels, val_pred),
            "test_metrics": _metrics(test_labels, test_pred),
        }
        (output_path.parent.parent / "image_training_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        return result
