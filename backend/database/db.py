from __future__ import annotations

import sqlite3
import json
from pathlib import Path

from backend.utils.config import DATABASE_URL, ROOT_DIR

DB_PATH = Path(DATABASE_URL.replace("sqlite:///", "", 1))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    connection = _connect()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                severity TEXT DEFAULT 'LOW',
                growth_probability REAL DEFAULT 0.0,
                latitude REAL,
                longitude REAL,
                affected_area REAL DEFAULT 0.0,
                image_path TEXT,
                confidence_percentage REAL DEFAULT 0.0,
                bounding_boxes TEXT DEFAULT '[]',
                location_status TEXT DEFAULT 'Location unavailable',
                alert_status TEXT DEFAULT 'NO ALERT',
                model_used TEXT DEFAULT 'YOLO + EfficientNet + LSTM',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                severity TEXT DEFAULT 'LOW',
                confidence REAL DEFAULT 0.0,
                growth_probability REAL DEFAULT 0.0,
                detection_type TEXT DEFAULT 'no_fire_smoke',
                alert_status TEXT DEFAULT 'NO ALERT',
                latitude REAL,
                longitude REAL,
                image_path TEXT,
                affected_area REAL DEFAULT 0.0,
                message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL,
                longitude REAL,
                place_name TEXT,
                source TEXT DEFAULT 'unknown'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL DEFAULT 0.0,
                precision REAL DEFAULT 0.0,
                recall REAL DEFAULT 0.0,
                f1_score REAL DEFAULT 0.0,
                map_score REAL DEFAULT 0.0,
                inference_time REAL DEFAULT 0.0,
                status TEXT DEFAULT 'Not evaluated',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        detection_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(detections)")
        }
        for name, definition in {
            "confidence_percentage": "REAL DEFAULT 0.0",
            "bounding_boxes": "TEXT DEFAULT '[]'",
            "location_status": "TEXT DEFAULT 'Location unavailable'",
            "alert_status": "TEXT DEFAULT 'NO ALERT'",
        }.items():
            if name not in detection_columns:
                connection.execute(f"ALTER TABLE detections ADD COLUMN {name} {definition}")

        alert_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(alerts)")
        }
        for name, definition in {
            "detection_type": "TEXT DEFAULT 'no_fire_smoke'",
            "alert_status": "TEXT DEFAULT 'NO ALERT'",
        }.items():
            if name not in alert_columns:
                connection.execute(f"ALTER TABLE alerts ADD COLUMN {name} {definition}")
        connection.commit()
    finally:
        connection.close()


def get_db():
    connection = _connect()
    try:
        yield connection
    finally:
        connection.close()


def _num(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def save_detection(result: dict, image_path: str) -> None:
    growth = result.get("growth") or {}
    overall = result.get("overall_risk") or result.get("risk") or {}
    detection = str(result.get("detection") or "unavailable")
    # Preserve None (NULL) for unavailable numerics — never fake 0.0 scores.
    raw_conf = result.get("confidence")
    try:
        confidence = float(raw_conf) if raw_conf is not None else None
    except (TypeError, ValueError):
        confidence = None
    # Severity: keep honest strings; new contract sends {"level","score","method"}.
    raw_sev = result.get("severity")
    if isinstance(raw_sev, dict):
        raw_sev = raw_sev.get("level")
    severity = str(raw_sev) if raw_sev not in (None, "") else "Severity Unavailable"
    growth_probability = result.get("growth_probability")
    if growth_probability is None:
        growth_probability = growth.get("probability")
    try:
        growth_probability = float(growth_probability) if growth_probability is not None else None
    except (TypeError, ValueError):
        growth_probability = None
    latitude = result.get("latitude")
    longitude = result.get("longitude")
    affected = result.get("affected_area")
    if affected is None:
        affected = (result.get("affected_area_result") or {}).get("value")
    try:
        affected_area = float(affected) if affected is not None else None
    except (TypeError, ValueError):
        affected_area = None
    raw_pct = result.get("confidence_percentage")
    try:
        confidence_percentage = float(raw_pct) if raw_pct is not None else (confidence * 100.0 if confidence is not None else None)
    except (TypeError, ValueError):
        confidence_percentage = None
    bounding_boxes = result.get("bounding_boxes") or result.get("annotations") or []
    location_status = str(result.get("location_status") or "Location unavailable")
    alert_status = str(result.get("alert_status") or "NO ALERT")
    model_used = str(result.get("model_used") or "EfficientNet + evidence")
    timestamp = str(result.get("timestamp") or "")
    connection = _connect()
    try:
        connection.execute(
            """
            INSERT INTO detections (
                detection_type, confidence, severity, growth_probability,
                latitude, longitude, affected_area, image_path,
                confidence_percentage, bounding_boxes, location_status,
                alert_status, model_used, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection, confidence, severity,
                growth_probability, latitude, longitude,
                affected_area, image_path, confidence_percentage,
                json.dumps(bounding_boxes), location_status,
                alert_status, model_used, timestamp,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def save_alert(alert: dict) -> None:
    connection = _connect()
    try:
        growth = alert.get("growth_probability")
        try:
            growth = float(growth) if growth is not None else None
        except (TypeError, ValueError):
            growth = None
        raw_conf = alert.get("confidence")
        try:
            conf = float(raw_conf) if raw_conf is not None else None
        except (TypeError, ValueError):
            conf = None
        raw_sev = alert.get("severity")
        sev = str(raw_sev) if raw_sev not in (None, "") else "Severity Unavailable"
        raw_aff = alert.get("affected_area")
        try:
            aff = float(raw_aff) if raw_aff is not None else None
        except (TypeError, ValueError):
            aff = None
        connection.execute(
            """
            INSERT INTO alerts (
                title, severity, confidence, growth_probability,
                detection_type, alert_status, latitude, longitude,
                affected_area, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(alert.get("title") or "WILDFIRE ALERT"),
                sev,
                conf,
                growth,
                str(alert.get("detection") or "unavailable"),
                str(alert.get("alert_status") or "NO ALERT"),
                alert.get("latitude"),
                alert.get("longitude"),
                aff,
                alert.get("message"),
                str(alert.get("created_at") or ""),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def list_saved_alerts() -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT id, title, detection_type AS detection, severity, confidence,
                   growth_probability, latitude, longitude, affected_area,
                   message, alert_status, created_at
            FROM alerts
            ORDER BY id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def list_detections(limit: int = 50) -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT id, detection_type AS detection, confidence, severity,
                   growth_probability, latitude, longitude, affected_area,
                   image_path, confidence_percentage, bounding_boxes,
                   location_status, alert_status, model_used, timestamp
            FROM detections
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def count_detections() -> int:
    connection = _connect()
    try:
        row = connection.execute("SELECT COUNT(*) AS c FROM detections").fetchone()
        return int(row["c"]) if row else 0
    finally:
        connection.close()


def get_detection(detection_id: int) -> dict | None:
    connection = _connect()
    try:
        row = connection.execute(
            """
            SELECT id, detection_type AS detection, confidence, severity,
                   growth_probability, latitude, longitude, affected_area,
                   image_path, confidence_percentage, bounding_boxes,
                   location_status, alert_status, model_used, timestamp
            FROM detections
            WHERE id = ?
            """,
            (int(detection_id),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def delete_detection(detection_id: int) -> bool:
    connection = _connect()
    try:
        cursor = connection.execute(
            "DELETE FROM detections WHERE id = ?", (int(detection_id),)
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()
