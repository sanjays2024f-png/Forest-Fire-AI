from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai.training.image_features import extract_image_features


def evaluate_image_model(model_path: str | Path, image_root: str | Path) -> Dict[str, object]:
    if not Path(model_path).is_file():
        return {"status": "not_evaluated", "reason": f"Model not found: {model_path}"}
    model = joblib.load(model_path)
    root = Path(image_root)
    paths: List[Path] = []
    labels: List[str] = []
    for class_dir in sorted(root.iterdir()) if root.exists() else []:
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(path)
                labels.append(class_dir.name)

    if not paths:
        return {"status": "not_evaluated", "reason": f"No images found in {root}"}

    features = np.vstack([extract_image_features(path) for path in paths])
    predictions = model.predict(features)
    result: Dict[str, object] = {
        "status": "evaluated",
        "sample_count": len(labels),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(labels, predictions, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(model.classes_)).tolist(),
    }
    if hasattr(model, "predict_proba") and len(model.classes_) == 2:
        probabilities = model.predict_proba(features)[:, 1]
        result["roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 4)
    else:
        result["roc_auc"] = None
    return result
