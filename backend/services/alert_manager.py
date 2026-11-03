from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional


class AlertManager:
    """Manage demo-mode and production alert records."""

    def __init__(self) -> None:
        self.alert_history: List[Dict] = []

    def build_alert(self, result: Dict, threshold: float = 0.75) -> Optional[Dict]:
        if not result.get("alert"):
            return None

        timestamp = result.get("timestamp") or datetime.now(timezone.utc).isoformat()
        raw_det = result.get("detection")
        # New contract uses uppercase prediction; accept both forms.
        pred = result.get("prediction")
        det_map = {"FIRE": "fire", "SMOKE": "smoke", "NO_FIRE": "no_fire_smoke",
                   "UNCERTAIN": "uncertain", "MODEL NOT TRAINED": "unavailable"}
        detection = det_map.get(str(pred), None) if pred else None
        if detection is None:
            detection = str(raw_det) if raw_det not in (None, "") else "unavailable"
        sev_obj = result.get("severity")
        if isinstance(sev_obj, dict):
            severity = str(sev_obj.get("level") or "Severity Unavailable")
        else:
            severity = str(sev_obj) if sev_obj not in (None, "") else "Severity Unavailable"
        raw_conf = result.get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None else None
        except (TypeError, ValueError):
            confidence = None
        # Deduplicate repeated identical alerts: same event+severity within run.
        signature = (detection, severity)
        for existing in reversed(self.alert_history[-8:]):
            if (existing.get("detection"), existing.get("severity")) == signature:
                try:
                    prev = existing.get("confidence")
                    if prev is None or confidence is None:
                        return None
                    if abs(float(prev) - confidence) < 0.02:
                        return None
                except (TypeError, ValueError):
                    return None
        raw_area = result.get("affected_area")
        try:
            area = float(raw_area) if raw_area is not None else None
        except (TypeError, ValueError):
            area = None
        raw_growth = result.get("growth_probability")
        if raw_growth is None:
            growth_obj = result.get("growth")
            raw_growth = growth_obj.get("probability") if isinstance(growth_obj, dict) else None
        try:
            growth_p = float(raw_growth) if raw_growth is not None else None
        except (TypeError, ValueError):
            growth_p = None
        if detection == "uncertain":
            message = (
                f"Uncertain classification (top confidence "
                f"{confidence * 100:.1f}%). Human review recommended."
                if confidence is not None else
                "Uncertain classification. Human review recommended."
            )
        elif confidence is not None:
            message = (
                f"{detection} detected with "
                f"{confidence * 100:.1f}% model confidence. "
                f"AI-generated, unverified until confirmed."
            )
        else:
            message = f"{detection} detected (model confidence unavailable). AI-generated, unverified."
        alert = {
            "id": len(self.alert_history) + 1,
            "title": "EARLY WILDFIRE WARNING",
            "detection": detection,
            "severity": severity,
            "confidence": confidence,
            "growth_probability": growth_p,
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "affected_area": area,
            "message": message,
            "alert_status": result.get("alert_status", "NO ALERT"),
            "created_at": timestamp,
            "threshold": threshold,
        }
        self.alert_history.append(alert)
        return alert

    def list_alerts(self) -> List[Dict]:
        return list(self.alert_history)
