from __future__ import annotations

from ai.inference.pipeline import HybridForestFireInferencePipeline


def main() -> None:
    pipeline = HybridForestFireInferencePipeline()
    sample = pipeline.predict("demo_input.jpg", source_type="image")
    print("Sample inference output:")
    print(sample)
    print("\nThis demo output is not real AI accuracy. It is intended for demonstration only.")


if __name__ == "__main__":
    main()
