from __future__ import annotations

from typing import Any, Dict, List

from ai.efficientnet.efficientnet_classifier import EfficientNetClassifier
from ai.lstm.lstm_temporal import LSTMWildfireRiskPredictor
from ai.yolo.yolo_detector import YoloFireSmokeDetector


class HybridForestFireInferencePipeline:
    """Honest inference entrypoint (kept for research scripts; API uses DemoForestFirePipeline).

    Never fabricates sequences or confidences. Without checkpoints it returns
    INSUFFICIENT_DATA / unavailable markers.
    """

    def __init__(self, yolo_model: str | None = None, efficientnet_model: str | None = None, lstm_model: str | None = None):
        self.detector = YoloFireSmokeDetector(yolo_model)
        self.classifier = EfficientNetClassifier(efficientnet_model)
        self.temporal = LSTMWildfireRiskPredictor(lstm_model)

    def predict(self, image_or_path: Any, source_type: str = "image") -> Dict[str, Any]:
        detections = self.detector.detect(image_or_path) if self.detector.available else []
        # No fake temporal sequence: single image -> INSUFFICIENT_DATA.
        # LSTM is used only with real ordered sequences (>=2 frames).
        temporal = self.temporal.predict_sequence([])
        return {
            "detection": detections[0]["label"] if detections else "unavailable",
            "confidence": detections[0]["confidence"] if detections else None,
            "severity": "Severity Unavailable",
            "severity_score": None,
            "risk": {"score": None, "level": "Risk Unavailable"},
            "risk_score": None,
            "growth_probability": None,
            "growth_label": "INSUFFICIENT_DATA",
            "growth": {"probability": None, "level": "INSUFFICIENT_DATA",
                       "reason": "Single image; temporal sequence required (need >=2 real ordered frames)"},
            "detections": detections,
            "classifications": [],
            "temporal": temporal,
            "mode": "EFFICIENTNET" if self.classifier.available else "FALLBACK",
        }
