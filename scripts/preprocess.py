from __future__ import annotations

from pathlib import Path

from ai.preprocessing.preprocessor import DatasetPreprocessor


def main() -> None:
    dataset_root = Path(__file__).resolve().parents[1] / "dataset"
    preprocessor = DatasetPreprocessor(dataset_root)
    report = preprocessor.run_full_pipeline()
    print("Dataset preprocessing summary:")
    print(report)
    print("\nPlace your NASA Earthdata dataset under dataset/train, dataset/val, dataset/test.")
    print("If labels differ, map them into: fire, smoke, no_fire, cloud, fog, dust")


if __name__ == "__main__":
    main()
