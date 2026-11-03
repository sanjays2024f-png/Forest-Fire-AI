from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.services.alert_manager import AlertManager
from backend.database.db import (
    delete_detection,
    get_detection,
    list_saved_alerts,
    save_alert,
    save_detection,
)
from backend.services.demo_pipeline import DemoForestFirePipeline
from backend.services.satellite_service import find_nearby_observation
from backend.utils.config import ALERT_THRESHOLD, MAX_UPLOAD_MB, ROOT_DIR

router = APIRouter()
engine = DemoForestFirePipeline()
alert_manager = AlertManager()


def _safe_filename(filename: str) -> str:
    safe = Path(filename).name
    if not safe:
        return "upload.jpg"
    return safe.replace(" ", "_")


def _save_upload(file: UploadFile, directory_name: str = "predictions") -> str:
    output_dir = ROOT_DIR / "outputs" / directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(file.filename or "upload.jpg")
    destination = output_dir / safe_name
    data = file.file.read()
    destination.write_bytes(data)
    return str(destination)


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "AI-Based Real-Time Forest Fire and Smoke Monitoring System",
        "mode": engine.model_status.get("efficientnet", "unavailable"),
        "primary_model": "EfficientNet-B0 (image) + MODIS CSV classifier (satellite) + LSTM backend-only",
        "model_status": engine.model_status,
    }


@router.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """Expose loaded-model facts without implying that unavailable models are trained."""
    trained = engine.image_model is not None or engine.efficientnet.available
    return {
        "status": "trained" if trained else "not_trained",
        "architecture": "EfficientNet-B0 (if weights loaded) else scikit-learn MLP image classifier",
        "classes": ["fire", "smoke", "no_fire"],
        "class_order": ["fire", "smoke", "no_fire"],
        "input": {"size": "MLP 32x32 RGB (/255); EfficientNet-B0 224x224 ImageNet norm",
                  "color": "RGB", "format": "JPG/JPEG/PNG/WEBP"},
        "checkpoint": {"mlp": str(engine.image_model_path),
                       "efficientnet": str(engine.efficientnet_path) if str(engine.efficientnet_path) not in ("", ".") else None,
                       "yolo": str(engine.yolo_path) if str(engine.yolo_path) not in ("", ".") else None,
                       "lstm": str(engine.lstm_path) if str(engine.lstm_path) not in ("", ".") else None},
        "model_status": engine.model_status,
        "model_state": "trained" if trained else "not_trained",
        "growth": {
            "status": "INSUFFICIENT_TEMPORAL_DATA",
            "reason": "Single image cannot support temporal forecasting; LSTM needs >=2 real ordered frames. TEMPORAL MODEL NOT TRAINED.",
        },
    }


@router.get("/model_info")
async def model_info_alias() -> Dict[str, Any]:
    return await model_info()


@router.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if Path(file.filename).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG, and WEBP images are supported.")
    if file.content_type and file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG, and WEBP images are supported.")
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=400, detail="Latitude and longitude must be provided together.")
    if latitude is not None and not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(status_code=400, detail="Latitude or longitude is outside its valid range.")

    saved_path = _save_upload(file, "predictions")
    try:
        size_mb = Path(saved_path).stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            Path(saved_path).unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_MB:g}MB limit.")
        with Image.open(saved_path) as image:
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        Path(saved_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    result = engine.predict_file(saved_path, latitude=latitude, longitude=longitude)
    satellite = find_nearby_observation(result.get("latitude"), result.get("longitude"))
    result["satellite"] = satellite
    # NOTE: engine.predict_file already returns the full API contract
    # (uppercase prediction, probabilities, severity/risk objects, growth,
    # location, affected_area, model_status, evidence). Do NOT overwrite
    # "prediction" with the lowercase internal detection label here.
    # Backend-calculated fields already include event/confidence/severity/risk.
    try:
        save_detection(result, saved_path)
    except Exception:
        pass
    alert = alert_manager.build_alert(result, threshold=ALERT_THRESHOLD)
    if alert:
        try:
            # Deduplicate identical stored alerts (same detection+severity+rounded conf).
            # None-safe: unavailable numerics never fabricate 0.0 or crash.
            def _rc(v):
                try:
                    return round(float(v), 2) if v is not None else None
                except (TypeError, ValueError):
                    return None
            new_conf = _rc(alert.get("confidence"))
            existing = list_saved_alerts()[:8]
            dup = any(
                (e.get("detection"), e.get("severity")) == (alert.get("detection"), alert.get("severity"))
                and _rc(e.get("confidence")) == new_conf
                for e in existing
            )
            if not dup:
                save_alert(alert)
                result["alert_record"] = alert
        except Exception:
            pass
    return result


# ---- Required aliases (do NOT break working API) ----
# Frontend upload request contains only `file`; backend calculates the rest.
@router.post("/upload-image")
async def upload_image_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await detect_image(file=file)


@router.post("/predict")
async def predict_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await detect_image(file=file)


@router.post("/predict-sequence")
async def predict_sequence_unavailable() -> Dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Temporal prediction is unavailable: no validated sequence model is loaded.",
    )


@router.post("/analyze-image")
async def analyze_image_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await detect_image(file=file)


@router.get("/history")
async def history(limit: int = 50) -> List[Dict[str, Any]]:
    from backend.database.db import list_detections
    try:
        return list_detections(limit=max(1, min(int(limit), 200)))
    except Exception:
        return []


@router.get("/history/{detection_id}")
async def history_item(detection_id: int) -> Dict[str, Any]:
    from backend.database.db import get_detection
    item = get_detection(detection_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return item


@router.delete("/history/{detection_id}")
async def history_delete(detection_id: int) -> Dict[str, Any]:
    from backend.database.db import delete_detection
    if not delete_detection(detection_id):
        raise HTTPException(status_code=404, detail="Detection not found")
    return {"status": "deleted", "id": detection_id}


@router.post("/webcam-reset")
async def webcam_reset() -> Dict[str, Any]:
    alert_manager.alert_history.clear()
    return {"status": "reset", "temporal": "INSUFFICIENT_DATA"}


@router.get("/temporal-status")
async def temporal_status() -> Dict[str, Any]:
    return {
        "available": False,
        "status": "INSUFFICIENT_DATA",
        "reason": "Single image; temporal sequence required. LSTM stays backend-only.",
        "lstm": engine.model_status.get("temporal"),
    }


@router.get("/api/satellite/training-report")
async def satellite_training_report() -> Dict[str, Any]:
    import json as _json
    report_path = ROOT_DIR / "outputs" / "satellite_training_report.json"
    if report_path.is_file():
        try:
            return _json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"status": "not_trained",
            "reason": "Run py scripts/train_satellite.py on NASA FIRMS CSVs first.",
            "expected": "outputs/satellite_training_report.json"}

@router.get("/api/alerts")
async def list_alerts() -> List[Dict[str, Any]]:
    return list_saved_alerts()


@router.get("/api/satellite/nearby")
async def satellite_nearby(latitude: Optional[float] = None, longitude: Optional[float] = None) -> Dict[str, Any]:
    return find_nearby_observation(latitude, longitude)


@router.get("/api/alerts/{alert_id}")
async def get_alert(alert_id: int) -> Dict[str, Any]:
    alerts = list_saved_alerts()
    for item in alerts:
        if item["id"] == alert_id:
            return item
    raise HTTPException(status_code=404, detail="Alert not found")


@router.get("/api/statistics")
async def statistics() -> Dict[str, Any]:
    from backend.database.db import count_detections
    alerts = list_saved_alerts()
    try:
        total = count_detections()
    except Exception:
        total = len(alerts)
    high_risk = sum(1 for item in alerts if item.get("severity") in {"HIGH", "CRITICAL"})
    total_area = 0.0
    for item in alerts:
        try:
            total_area += float(item.get("affected_area") or 0)
        except (TypeError, ValueError):
            continue
    return {
        "total_detections": total,
        "active_fires": sum(1 for item in alerts if item.get("detection") == "fire"),
        "smoke_detections": sum(1 for item in alerts if item.get("detection") == "smoke"),
        "high_risk_events": high_risk,
        "critical_alerts": sum(1 for item in alerts if item.get("severity") == "CRITICAL"),
        "estimated_affected_area": round(total_area, 2),
        "area_unit": "image-space-percent",
        "demo_mode": False,
    }


@router.get("/api/model-performance")
async def model_performance() -> Dict[str, Any]:
    """Real metrics: image training report + satellite CSV report, else 'Not evaluated'."""
    import json as _json
    sat_path = ROOT_DIR / "outputs" / "satellite_training_report.json"
    satellite = {"status": "Not evaluated - run scripts/train_satellite.py",
                 "accuracy": "Not evaluated", "precision": "Not evaluated",
                 "recall": "Not evaluated", "f1": "Not evaluated"}
    try:
        if sat_path.is_file():
            trained = _json.loads(sat_path.read_text(encoding="utf-8"))
            if trained and trained.get("final_test"):
                ft = trained["final_test"]
                satellite = {"status": "trained on NASA FIRMS MODIS 2024 (sampled)",
                             "accuracy": ft.get("accuracy"), "precision": ft.get("precision"),
                             "recall": ft.get("recall"), "f1": ft.get("f1"),
                             "macro_f1": ft.get("macro_f1"), "scope": ft.get("scope")}
    except Exception:
        pass
    img_path = ROOT_DIR / "outputs" / "image_training_report.json"
    if img_path.is_file():
        try:
            rep = _json.loads(img_path.read_text(encoding="utf-8"))
            image = {"status": f"trained ({rep.get('sample_count')} images, "
                               f"train/val/test {rep.get('split')})",
                     "train": rep.get("train_metrics"), "val": rep.get("val_metrics"),
                     "test": rep.get("test_metrics"),
                     "checkpoint": rep.get("checkpoint_path"),
                     "class_order": rep.get("class_order")}
        except Exception:
            image = {"status": "Image classifier: " + str(engine.model_status.get("image_classifier"))}
    else:
        image = {"status": ("EfficientNet-B0 loaded" if "efficientnet-b0 loaded" in str(engine.model_status.get("efficientnet"))
                            else "Image classifier: " + str(engine.model_status.get("image_classifier"))),
                 "note": "Train with scripts/download_fire_dataset.py then scripts/train.py for real metrics."}
    temporal = {"status": ("TEMPORAL MODEL NOT TRAINED; single images report INSUFFICIENT_TEMPORAL_DATA"
                           if engine.temporal.loaded is False else "LSTM loaded (sequences only)")}
    return {
        "Image classifier (fire/smoke/no_fire)": image,
        "Satellite CSV classifier (MODIS confidence)": satellite,
        "LSTM (temporal)": temporal,
        "YOLO": {"status": engine.model_status.get("yolo")},
    }
