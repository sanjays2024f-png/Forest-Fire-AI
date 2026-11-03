from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class ModelEvaluator:
    """Evaluation module for model comparison and metric generation."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_models(self) -> Dict[str, str]:
        return {
            "YOLO + EfficientNet + LSTM": "Not evaluated",
            "CNN + BiLSTM + Attention": "Not evaluated",
            "Swin Transformer + CNN": "Not evaluated",
            "YOLO + ViT + LSTM": "Not evaluated",
            "ConvLSTM + Attention": "Not evaluated",
        }

    def generate_report(self) -> List[Dict[str, str]]:
        return [
            {"Model": name, "Accuracy": value, "Precision": value, "Recall": value, "F1": value, "mAP": value, "Inference Time": value}
            for name, value in self.evaluate_models().items()
        ]
