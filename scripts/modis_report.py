from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_files = sorted((root / "dataset" / "modis" / "2024").glob("*.csv"))
    rows = []
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append({
                        "confidence": float(row["confidence"]),
                        "frp": float(row.get("frp") or 0),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue

    high = sum(1 for row in rows if row["confidence"] >= 80)
    medium = sum(1 for row in rows if 50 <= row["confidence"] < 80)
    low = len(rows) - high - medium
    report = {
        "source": "MODIS 2024 real observations",
        "csv_file_count": len(csv_files),
        "observation_count": len(rows),
        "confidence_distribution": {"HIGH": high, "MEDIUM": medium, "LOW": low},
        "mean_confidence": round(mean(row["confidence"] for row in rows), 4) if rows else None,
        "mean_frp": round(mean(row["frp"] for row in rows), 4) if rows else None,
        "risk_method": "MODIS observed confidence / 100 (FRP reported separately, no blending)",
    }
    output = root / "outputs" / "modis_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved MODIS report to {output}")


if __name__ == "__main__":
    main()
