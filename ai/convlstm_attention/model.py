from __future__ import annotations

from typing import Any, Dict


class ConvLSTMAttention:
    """Optional video sequence model for early wildfire prediction."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def predict(self, sequence: Any) -> Dict[str, Any]:
        return {"label": None, "confidence": None, "status": "Not evaluated"}
