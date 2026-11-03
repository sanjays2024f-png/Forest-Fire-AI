from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import joblib
import numpy as np
from PIL import Image

from ai.training.image_features import extract_image_features
from backend.utils.config import UNCERTAIN_THRESHOLD

logger = logging.getLogger(__name__)


class DemoForestFirePipeline:
    """Image inference pipeline with explicit model availability.

    Honest rules:
    - EfficientNet-B0 probabilities ONLY when real weights load.
    - scikit-learn MLP probabilities ONLY when best_model.pkl loads.
    - Otherwise detection/confidence/risk/severity are Unavailable (None),
      never pixel-heuristic guesses.
    - YOLO boxes ONLY with a real checkpoint; growth is always
      INSUFFICIENT_DATA for single images (LSTM needs >=2 ordered frames).
    """

    def __init__(self) -> None:
        from ai.efficientnet.efficientnet_classifier import EfficientNetClassifier
        from ai.yolo.yolo_detector import YoloFireSmokeDetector
        from ai.lstm.lstm_temporal import LSTMWildfireRiskPredictor
        self.yolo_path = Path(os.getenv("YOLO_MODEL_PATH", ""))
        self.efficientnet_path = Path(os.getenv("EFFICIENTNET_MODEL_PATH", ""))
        self.lstm_path = Path(os.getenv("LSTM_MODEL_PATH", ""))
        self.image_model_path = Path(
            os.getenv(
                "IMAGE_MODEL_PATH",
                str(Path(__file__).resolve().parents[2] / "outputs" / "models" / "best_model.pkl"),
            )
        )
        self.image_model = None
        if self.image_model_path.is_file():
            try:
                candidate = joblib.load(self.image_model_path)
                if hasattr(candidate, "predict_proba") and hasattr(candidate, "classes_"):
                    self.image_model = candidate
            except Exception:
                logger.exception("Unable to load image model from %s", self.image_model_path)
        # Honest wrappers: YOLO returns boxes ONLY with real checkpoint.
        # EfficientNet-B0 returns probs ONLY with real weights.
        self.yolo = YoloFireSmokeDetector(str(self.yolo_path) if self.yolo_path.is_file() else None)
        self.efficientnet = EfficientNetClassifier(str(self.efficientnet_path) if self.efficientnet_path.is_file() else None)
        self.temporal = LSTMWildfireRiskPredictor(str(self.lstm_path) if self.lstm_path.is_file() else None)
        self.model_status = {
            "yolo": "loaded" if self.yolo.available else "fallback (checkpoint unavailable)",
            "efficientnet": "efficientnet-b0 loaded" if self.efficientnet.available else "unavailable (weights not loaded)",
            "temporal": "LSTM loaded" if self.temporal.loaded else "LSTM unavailable - INSUFFICIENT_DATA",
            "image_classifier": "scikit-learn MLP loaded" if self.image_model is not None else "not trained",
            "satellite": "MODIS 2024",
        }
        logger.info("Model status: %s", self.model_status)

    @staticmethod
    def _extract_exif_location(file_path: str) -> Optional[Dict[str, float]]:
        with Image.open(file_path) as image:
            gps = image.getexif().get(34853)
            if not gps:
                return None

            def to_decimal(value: Any, reference: str) -> float:
                degrees, minutes, seconds = [float(part) for part in value]
                decimal = degrees + minutes / 60 + seconds / 3600
                return -decimal if reference in {"S", "W"} else decimal

            return {
                "latitude": to_decimal(gps[2], gps.get(1, "N")),
                "longitude": to_decimal(gps[4], gps.get(3, "E")),
            }

    def _pixel_fallback(self, file_path: str) -> Dict[str, Any]:
        """No classifier available: report dimensions only, NEVER fake probs.

        The old warm/smoke pixel heuristic (fire_ratio*4.0 etc.) was an
        arbitrary formula, not a trained model. Per honesty rules it must not
        produce Fire/Smoke/No-Fire probabilities, confidence, risk or severity.
        Callers must treat this backend as untrained (Risk/Severity Unavailable).
        """
        with Image.open(file_path) as _im:
            width, height = _im.size
        return {
            "label": "unavailable",
            "confidence": None,
            "probabilities": {"fire": None, "smoke": None, "no_fire_smoke": None},
            "boxes": [],
            "width": width,
            "height": height,
        }

    def _sklearn_prediction(self, file_path: str) -> Optional[Dict[str, Any]]:
        if self.image_model is None:
            return None
        try:
            features = extract_image_features(file_path).reshape(1, -1)
            probabilities = self.image_model.predict_proba(features)[0]
        except Exception:
            return None
        scores = {
            str(lb): max(0.0, float(s))
            for lb, s in zip(self.image_model.classes_, probabilities)
            if np.isfinite(float(s))
        }
        # Normalise to fire/smoke/no_fire_smoke keys.
        norm = {"fire": float(scores.get("fire", 0.0)),
                "smoke": float(scores.get("smoke", 0.0)),
                "no_fire_smoke": float(scores.get("no_fire_smoke", scores.get("no_fire", 0.0)))}
        total = sum(norm.values())
        if not np.isfinite(total) or total <= 0:
            return None
        norm = {k: round(v / total, 4) for k, v in norm.items()}
        norm["no_fire_smoke"] = round(
            max(0.0, norm["no_fire_smoke"] + round(1.0 - sum(norm.values()), 4)), 4,
        )
        label = max(norm, key=norm.get)
        confidence = norm[label]
        if confidence < UNCERTAIN_THRESHOLD:
            label = "uncertain"
        try:
            with Image.open(file_path) as _im:
                _w, _h = _im.size
        except Exception:
            _w, _h = 0, 0
        return {"label": label, "confidence": confidence, "probabilities": norm,
                "boxes": [], "width": _w, "height": _h, "backend": "sklearn-mlp"}

    SEVERITY_METHOD = (
        "Transparent operational rule on trained-classifier evidence only. "
        "score = 0.40*confidence + 0.40*event_likelihood*(1.0 fire | 0.75 smoke) "
        "+ 0.20*min(1, fire_area_pct/25). Levels CRITICAL>=0.80 HIGH>=0.60 "
        "MEDIUM>=0.40 else LOW. Confidence is model certainty, NOT severity; "
        "area is image-space % (no segmentation model); coordinates only from "
        "EXIF/user/verified source."
    )
    RISK_METHOD = (
        "Operational event-likelihood estimate: risk = 1 - P(no_fire). "
        "Levels HIGH>=0.65 MEDIUM>=0.40 else LOW. "
        "Disclosed operational buckets, not a calibrated model."
    )

    @staticmethod
    def _severity(
        label: str,
        affected_ratio: Optional[float] = None,
        probs: Optional[Dict[str, float]] = None,
        backend_trained: bool = False,
        confidence: Optional[float] = None,
    ) -> tuple[str, float | None]:
        """Transparent severity from real classifier evidence (no fake scale).

        - No trained backend -> "Severity Unavailable".
        - unavailable/uncertain labels stay unavailable/UNCERTAIN.
        - no_fire_smoke (trained) -> LOW / 0.0.
        - fire/smoke (trained) -> weighted operational score (see SEVERITY_METHOD).
        """
        probs = probs or {}
        if not backend_trained or label == "unavailable":
            return "Severity Unavailable", None
        if label == "uncertain":
            return "UNCERTAIN", None
        if label == "no_fire_smoke":
            return "LOW", 0.0
        try:
            p_fire = float(probs.get("fire") or 0.0)
            p_smoke = float(probs.get("smoke") or 0.0)
        except (TypeError, ValueError):
            p_fire, p_smoke = 0.0, 0.0
        try:
            conf = float(confidence) if confidence is not None else max(p_fire, p_smoke)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            area_pct = float(affected_ratio) if affected_ratio is not None else 0.0
        except (TypeError, ValueError):
            area_pct = 0.0
        p_event = max(0.0, min(1.0, p_fire + p_smoke))
        class_weight = 1.0 if label == "fire" else 0.75
        area_term = min(1.0, max(0.0, area_pct) / 25.0)
        score = 0.40 * max(0.0, min(1.0, conf)) + 0.40 * p_event * class_weight + 0.20 * area_term
        score = round(max(0.0, min(1.0, score)), 4)
        if score >= 0.80:
            return "CRITICAL", score
        if score >= 0.60:
            return "HIGH", score
        if score >= 0.40:
            return "MEDIUM", score
        return "LOW", score

    @staticmethod
    def _risk(
        probs: Optional[Dict[str, float]] = None,
        backend_trained: bool = False,
    ) -> tuple[str, float | None]:
        """Operational event-likelihood risk from real probabilities.

        risk = 1 - P(no_fire). Untrained backend -> Risk Unavailable.
        """
        if not backend_trained:
            return "Risk Unavailable", None
        try:
            p_no = (probs or {}).get("no_fire_smoke")
            if p_no is None:
                return "Risk Unavailable", None
            p_event = max(0.0, min(1.0, 1.0 - float(p_no)))
        except (TypeError, ValueError):
            return "Risk Unavailable", None
        score = round(p_event, 4)
        level = "HIGH" if p_event >= 0.65 else "MEDIUM" if p_event >= 0.4 else "LOW"
        return level, score

    def _apply_confidence_threshold(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """UNCERTAIN when top softmax < UNCERTAIN_THRESHOLD (configurable).

        Probabilities are preserved; only the label becomes 'uncertain'.
        Untrained backends stay 'unavailable' (MODEL NOT TRAINED) — never UNCERTAIN.
        """
        from backend.utils.config import UNCERTAIN_THRESHOLD
        try:
            conf = float(analysis.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if analysis.get("backend") in {"efficientnet-b0", "sklearn-mlp"} and conf < UNCERTAIN_THRESHOLD:
            analysis = dict(analysis)
            analysis["label"] = "uncertain"
        return analysis

    def predict_file(
        self,
        file_path: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        # BRANCH: EfficientNet-B0 first, then sklearn MLP, then pixel evidence.
        # CSV/satellite model is NEVER used for image classification.
        analysis = None
        backend_used = "pixel-evidence"
        try:
            eff = self.efficientnet.predict_image(file_path)
            if eff:
                probs = eff.get("probabilities") or {}
                norm = {"fire": float(probs.get("fire", 0.0)),
                        "smoke": float(probs.get("smoke", 0.0)),
                        "no_fire_smoke": float(probs.get("no_fire", probs.get("no_fire_smoke", 0.0)))}
                tot = sum(norm.values()) or 1.0
                norm = {k: round(v / tot, 4) for k, v in norm.items()}
                raw = str(eff.get("label", "no_fire")).lower()
                lbl = "no_fire_smoke" if raw in {"no_fire", "no_fire_smoke"} else raw
                with Image.open(file_path) as _im:
                    _w, _h = _im.size
                analysis = {"label": lbl, "confidence": float(eff.get("confidence", norm[lbl])),
                            "probabilities": norm, "width": _w, "height": _h, "backend": "efficientnet-b0"}
                backend_used = "efficientnet-b0"
        except Exception:
            analysis = None
        if analysis is None:
            analysis = self._sklearn_prediction(file_path)
            if analysis is not None:
                backend_used = "sklearn-mlp"
        if analysis is None:
            analysis = self._pixel_fallback(file_path)
            backend_used = "pixel-evidence"
        # Real YOLO boxes ONLY if checkpoint loaded; never fake boxes.
        # Without a detector, show the original image: boxes stay empty and
        # affected area stays Unavailable (no fabricated overlay boxes).
        yolo_boxes: List[Dict[str, Any]] = []
        try:
            yolo_boxes = self.yolo.detect(file_path) if self.yolo.available else []
        except Exception:
            yolo_boxes = []
        boxes: List[Dict[str, Any]] = []
        if yolo_boxes:
            # Normalize YOLO xyxy bbox -> 0..1 width/height for area + UI overlay.
            img_w = max(1, int(analysis.get("width") or 1))
            img_h = max(1, int(analysis.get("height") or 1))
            for det in yolo_boxes:
                try:
                    x1, y1, x2, y2 = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
                    w = max(0.0, min(1.0, (x2 - x1) / img_w))
                    h = max(0.0, min(1.0, (y2 - y1) / img_h))
                    x = max(0.0, min(1.0, x1 / img_w))
                    y = max(0.0, min(1.0, y1 / img_h))
                except Exception:
                    continue
                boxes.append({"label": str(det.get("label", analysis["label"])),
                              "confidence": float(det.get("confidence", analysis["confidence"])),
                              "x": round(x, 4), "y": round(y, 4),
                              "width": round(w, 4), "height": round(h, 4),
                              "bbox": det.get("bbox")})
        # Affected ratio from real YOLO boxes only; None when no boxes (no fabrication).
        if boxes:
            affected_ratio: float | None = min(1.0, sum(float(b.get("width", 0)) * float(b.get("height", 0)) for b in boxes))
            area_value: Optional[float] = round(affected_ratio * 100, 2)
        else:
            affected_ratio = None
            area_value = None
        backend_trained = backend_used in {"efficientnet-b0", "sklearn-mlp"}
        if backend_trained:
            analysis = self._apply_confidence_threshold(analysis)
        severity, severity_score = self._severity(analysis["label"], affected_ratio,
                                                  analysis.get("probabilities") or {},
                                                  backend_trained=backend_trained,
                                                  confidence=analysis.get("confidence"))
        exif_location = self._extract_exif_location(file_path) if latitude is None or longitude is None else None
        resolved_latitude = latitude if latitude is not None else (exif_location or {}).get("latitude")
        resolved_longitude = longitude if longitude is not None else (exif_location or {}).get("longitude")
        location_status = "GPS metadata" if exif_location and latitude is None and longitude is None else (
            "Manual coordinates" if latitude is not None and longitude is not None else "Location unavailable"
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        label = analysis["label"]
        # RISK from actual model predictions only — no arbitrary blends.
        # confidence = P(predicted class); risk = P(fire OR smoke) = 1 - P(no_fire).
        # These are distinct numbers by construction. When the image classifier
        # is NOT trained (pixel-evidence fallback) there are no reliable
        # probabilistic inputs -> Risk Unavailable (score None), never 0.0 fake.
        probs = analysis.get("probabilities") or {}
        risk_level, risk_score = self._risk(probs, backend_trained=backend_trained)
        # UNCERTAIN predictions carry risk but never trigger alerts on their own.
        from backend.utils.config import ALERT_THRESHOLD
        try:
            p_ev = 1.0 - float(probs.get("no_fire_smoke")) if probs.get("no_fire_smoke") is not None else None
        except (TypeError, ValueError):
            p_ev = None
        alert = bool(backend_trained and label in {"fire", "smoke"} and p_ev is not None and p_ev >= ALERT_THRESHOLD)
        sev_log = f"{severity} {severity_score:.4f}" if severity_score is not None else f"{severity} n/a"
        risk_log = f"{risk_level} {risk_score:.4f}" if risk_score is not None else f"{risk_level} n/a"
        conf_log = f"{analysis['confidence']:.4f}" if analysis.get("confidence") is not None else "n/a"
        logger.info(
            "[IMAGE] file=%s size=%sx%s backend=%s; [PROBS] %s; [SEV] %s [RISK] %s [CONF] %s",
            Path(file_path).name, analysis["width"], analysis["height"], backend_used,
            analysis["probabilities"], sev_log, risk_log, conf_log,
        )

        def _r4(v: Any) -> float | None:
            try:
                return round(float(v), 4) if v is not None else None
            except (TypeError, ValueError):
                return None

        safe_probs = {k: _r4(v) for k, v in (analysis.get("probabilities") or {}).items()}
        conf4 = _r4(analysis.get("confidence"))
        pred_upper = {"fire": "FIRE", "smoke": "SMOKE", "no_fire_smoke": "NO_FIRE",
                      "uncertain": "UNCERTAIN", "unavailable": "MODEL NOT TRAINED"}.get(label, "MODEL NOT TRAINED")
        model_state = "trained" if backend_trained else "not_trained"
        evidence = []
        if backend_trained:
            evidence.append(f"{backend_used} softmax probabilities: {safe_probs}")
            evidence.append(f"severity rule: {self.SEVERITY_METHOD}")
            evidence.append(f"risk rule: {self.RISK_METHOD}")
            if boxes:
                evidence.append(f"{len(boxes)} YOLO box(es); image-space area {area_value}%")
            else:
                evidence.append("no detector boxes; area unavailable (no segmentation model)")
        else:
            evidence.append("no trained image checkpoint loaded (MODEL NOT TRAINED)")

        return {
            # API contract (values from live inference above):
            "prediction": pred_upper,
            "confidence": conf4,
            "confidence_percentage": round(conf4 * 100, 2) if conf4 is not None else None,
            "probabilities": {"fire": safe_probs.get("fire"), "smoke": safe_probs.get("smoke"),
                              "no_fire": safe_probs.get("no_fire_smoke")},
            "severity": {"level": severity, "score": severity_score, "method": self.SEVERITY_METHOD},
            "severity_level": severity,
            "severity_score": severity_score,
            "risk": {"score": risk_score, "level": risk_level, "method": self.RISK_METHOD},
            "risk_score": risk_score,
            "growth": {"probability": None, "level": "INSUFFICIENT_TEMPORAL_DATA",
                       "status": "INSUFFICIENT_TEMPORAL_DATA",
                       "message": "A reliable temporal forecast requires suitable sequential observations.",
                       "reason": "Single image; temporal sequence required (need >=2 real ordered frames). LSTM used only with real ordered sequences."},
            "growth_probability": None,
            "growth_label": "INSUFFICIENT_DATA",
            "location": ({"latitude": resolved_latitude, "longitude": resolved_longitude,
                           "status": location_status}
                          if resolved_latitude is not None and resolved_longitude is not None else None),
            "latitude": resolved_latitude,
            "longitude": resolved_longitude,
            "location_status": location_status,
            "affected_area": area_value,
            "estimated_affected_area": area_value,
            "area_unit": "image-space-percent" if area_value is not None else "unavailable",
            "model_status": model_state,
            "model_details": self.model_status,
            "model_used": (
                "EfficientNet-B0" if backend_used == "efficientnet-b0"
                else "Scikit-learn MLP image classifier" if backend_used == "sklearn-mlp"
                else "MODEL NOT TRAINED"
            ),
            "evidence": evidence,
            # Compatibility keys:
            "detection": label,
            "class_probabilities": safe_probs,
            "bounding_boxes": boxes,
            "timestamp": timestamp,
            "alert": alert,
            "alert_status": "HIGH-RISK WILDFIRE DETECTED" if alert else "NO ALERT",
            "mode": "EFFICIENTNET" if backend_used == "efficientnet-b0" else "FALLBACK",
            "detection_result": {"type": label, "confidence": conf4},
            "severity_result": {"level": severity, "score": severity_score},
            "satellite": {"risk": None, "available": False},
            "affected_area_result": {"value": area_value, "unit": "percent" if area_value is not None else "unavailable", "type": "image-space"},
            "overall_risk": {"score": risk_score, "level": risk_level},
            "efficientnet": {"label": label, "confidence": conf4,
                             "probabilities": safe_probs,
                             "backend": backend_used},
            "alert_result": {"level": "HIGH-RISK WILDFIRE DETECTED" if alert else "NO ALERT", "message": None},
            "sources": [],
            "annotations": boxes,
        }
