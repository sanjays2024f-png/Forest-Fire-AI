from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.training.evaluator import evaluate_image_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "outputs" / "models" / "best_model.pkl"
    result = evaluate_image_model(model_path, root / "dataset" / "test")
    # Merge with training report when available for a complete picture.
    train_rep_path = root / "outputs" / "image_training_report.json"
    if train_rep_path.is_file():
        try:
            train_rep = json.loads(train_rep_path.read_text(encoding="utf-8"))
            result["train_split"] = train_rep.get("split")
            result["train_metrics"] = train_rep.get("train_metrics")
            result["val_metrics"] = train_rep.get("val_metrics")
            result["dataset_source"] = (json.loads((root / "outputs" / "dataset_report.json").read_text(encoding="utf-8")).get("source")
                                        if (root / "outputs" / "dataset_report.json").is_file() else None)
        except Exception:
            pass
    report_path = root / "outputs" / "image_evaluation_report.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
