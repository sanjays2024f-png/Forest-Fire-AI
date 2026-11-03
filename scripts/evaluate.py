from __future__ import annotations

from pathlib import Path

from ai.evaluation.evaluator import ModelEvaluator


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    evaluator = ModelEvaluator(output_dir)
    report = evaluator.generate_report()
    print("Model evaluation table:")
    print(report)
    print("\nNo trained models were found, so all models currently show: Not evaluated")


if __name__ == "__main__":
    main()
