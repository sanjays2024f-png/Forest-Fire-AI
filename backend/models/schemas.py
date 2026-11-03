from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: Literal["image"] = "image"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[datetime] = None
    file_name: Optional[str] = None


class DetectionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    detection: str
    # Honest nullable: untrained classifier -> None (Unavailable), never fake 0.0.
    confidence: Optional[float] = None
    confidence_percentage: Optional[float] = None
    bounding_boxes: List[dict] = Field(default_factory=list)
    severity: str = "Severity Unavailable"
    estimated_affected_area: Optional[float] = None
    location_status: str = "Location unavailable"
    alert_status: str = "NO ALERT"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    affected_area: Optional[float] = None
    area_unit: str = "unavailable"
    timestamp: str = ""
    alert: bool = False
    model_used: str = "Image classifier not trained"
    mode: str = "UNAVAILABLE"


class AlertEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    detection: str
    severity: str
    confidence: float
    growth_probability: float
    latitude: Optional[float]
    longitude: Optional[float]
    affected_area: float
    message: str
    alert_status: str
    created_at: str
