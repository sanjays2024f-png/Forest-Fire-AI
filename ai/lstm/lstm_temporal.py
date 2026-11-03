from __future__ import annotations
from typing import Any, Dict, List
class LSTMWildfireRiskPredictor:
    """Honest temporal module. Single image -> INSUFFICIENT_DATA. No fake curves."""
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.loaded = False
        self._model = None
        if model_path:
            try:
                from pathlib import Path
                import joblib
                p = Path(model_path)
                if p.is_file():
                    self._model = joblib.load(p)
                    self.loaded = True
            except Exception:
                self._model = None
                self.loaded = False
    def predict_sequence(self, sequence: List[float]) -> Dict[str, Any]:
        if not sequence or len(sequence) < 2:
            return {"growth_probability": None, "risk_label": "INSUFFICIENT_DATA",
                    "status": "INSUFFICIENT_DATA",
                    "reason": "Single image; temporal sequence required"}
        if self._model is None:
            return {"growth_probability": None, "risk_label": "INSUFFICIENT_DATA",
                    "status": "INSUFFICIENT_DATA",
                    "reason": "LSTM checkpoint unavailable; real sequence present but no trained LSTM"}
        try:
            import numpy as np
            arr = np.asarray(sequence, dtype=float).reshape(1, -1)
            prob = float(self._model.predict(arr)[0] if hasattr(self._model, "predict") else 0.0)
            prob = max(0.0, min(1.0, prob))
            label = "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.4 else "LOW"
            return {"growth_probability": round(prob, 4), "risk_label": label, "status": "OK"}
        except Exception:
            return {"growth_probability": None, "risk_label": "INSUFFICIENT_DATA",
                    "status": "INSUFFICIENT_DATA", "reason": "LSTM inference failed"}
    def predict_frame_sequence(self, frames: List[Any]) -> Dict[str, Any]:
        if not frames or len(frames) < 2:
            return {"growth_probability": None, "risk_label": "INSUFFICIENT_DATA",
                    "status": "INSUFFICIENT_DATA",
                    "reason": "Temporal sequence required (need >=2 real frames)"}
        vals = []
        for f in frames:
            try:
                vals.append(float(f))
            except Exception:
                continue
        return self.predict_sequence(vals)

