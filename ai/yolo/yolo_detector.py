from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
class YoloFireSmokeDetector:
    """Honest YOLO wrapper: returns boxes ONLY if a real checkpoint is loaded."""
    def __init__(self, model_path: str | None = None):
        self.model_path = str(model_path or "")
        self.loaded = bool(self.model_path) and Path(self.model_path).is_file()
        self._model = None
        if self.loaded:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except Exception:
                self._model = None
                self.loaded = False
    @property
    def available(self) -> bool:
        return self._model is not None
    def detect(self, image_or_path: Any) -> List[Dict[str, Any]]:
        if self._model is None:
            return []
        try:
            res = self._model.predict(str(image_or_path), verbose=False)[0]
            out = []
            for b in res.boxes:
                out.append({"label": str(res.names[int(b.cls[0])]),
                    "confidence": round(float(b.conf[0]), 4),
                    "bbox": [round(float(v), 1) for v in b.xyxy[0].tolist()]})
            return out
        except Exception:
            return []
    def detect_file(self, file_path: str) -> List[Dict[str, Any]]:
        return self.detect(file_path)

