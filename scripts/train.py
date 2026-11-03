from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.training.trainer import TrainingEngine


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    engine = TrainingEngine()
    result = engine.train(dataset_dir)
    print("Training status:")
    print(result)
    if result.get("training_completed"):
        print("\nThe trained image classifier is now available to the API.")
    else:
        print("\nNo image model was trained. Add labeled images under dataset/train/<fire|smoke|no_fire>.")


if __name__ == "__main__":
    main()
