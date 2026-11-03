from __future__ import annotations

from typing import Any, Dict


class YoloViT_LSTM:
    """Optional YOLO + ViT + LSTM experimental model."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def predict(self, image: Any) -> Dict[str, Any]:
        return {"label": None, "confidence": None, "status": "Not evaluated"}
