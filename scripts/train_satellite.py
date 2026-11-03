from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.training.tabular_fire_pipeline import run_satellite_training
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("STEP 1-11: NASA FIRMS MODIS CSV -> leakage -> 80/20 -> 10-fold CV on 80%")
    print("         -> SMOTE training-only -> initial -> overfit check -> correction")
    print("         -> retrain -> FINAL 20% test. Source: https://firms.modaps.eosdis.nasa.gov/country/")
    report = run_satellite_training(root / "dataset", root / "outputs")
    print("\nDone. Report: outputs/satellite_training_report.json")
    print(f"FINAL TEST (20% unseen): {report.get('final_test')}")
if __name__ == "__main__":
    main()
