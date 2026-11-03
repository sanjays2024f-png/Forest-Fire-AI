from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

from backend.utils.config import ROOT_DIR

DATASET_ROOT = ROOT_DIR / "dataset" / "modis" / "2024"
_cache: Dict[tuple[float, float], Dict[str, Any]] = {}


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat_delta = math.radians(lat2 - lat1)
    lon_delta = math.radians(lon2 - lon1)
    value = (
        math.sin(lat_delta / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(lon_delta / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(value))


def find_nearby_observation(latitude: Optional[float], longitude: Optional[float]) -> Dict[str, Any]:
    if latitude is None or longitude is None:
        return {
            "available": False,
            "risk": None,
            "status": "Location required for satellite analysis",
            "source": "MODIS 2024",
        }

    cache_key = (round(latitude, 3), round(longitude, 3))
    if cache_key in _cache:
        return _cache[cache_key]

    nearest: Optional[Dict[str, Any]] = None
    nearest_distance = float("inf")
    for csv_path in DATASET_ROOT.glob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row_latitude = float(row["latitude"])
                    row_longitude = float(row["longitude"])
                    distance = _distance_km(latitude, longitude, row_latitude, row_longitude)
                except (KeyError, TypeError, ValueError):
                    continue
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = row

    if nearest is None or nearest_distance > 100:
        result = {
            "available": False,
            "risk": None,
            "status": "Satellite data unavailable",
            "source": "MODIS 2024",
        }
    else:
        confidence = float(nearest.get("confidence") or 0)
        # Honest mapping: MODIS 'confidence' is the real observation signal.
        # Report it directly (confidence/100); do NOT blend FRP with arbitrary
        # 70/30 weights. Level buckets are disclosed operational mappings of
        # the observed confidence value, not a calibrated risk model.
        risk = "HIGH" if confidence >= 80 else "MEDIUM" if confidence >= 50 else "LOW"
        frp = float(nearest.get("frp") or 0)
        risk_score = round(min(1.0, confidence / 100.0), 4)
        result = {
            "available": True,
            "risk": risk,
            "risk_score": risk_score,
            "risk_method": "MODIS observed confidence / 100 (FRP reported separately, no blending)",
            "status": f"{risk} MODIS signal",
            "source": "MODIS 2024 real observation",
            "distance_km": round(nearest_distance, 2),
            "observation": {
                "latitude": float(nearest["latitude"]),
                "longitude": float(nearest["longitude"]),
                "confidence": confidence,
                "frp": frp,
                "acq_date": nearest.get("acq_date"),
                "satellite": nearest.get("satellite"),
            },
        }

    _cache[cache_key] = result
    return result
