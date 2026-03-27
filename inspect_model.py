from __future__ import annotations

import argparse
from pathlib import Path

from own_ai_model import format_artifact_summary, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved model artifact and print its metadata.")
    parser.add_argument(
        "--model",
        default="artifacts/model_params.json",
        help="Path to the trained model parameters JSON file.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path.resolve()}")

    _, payload = load_model(model_path)
    print(f"Model file: {model_path.resolve()}")
    print(format_artifact_summary(payload))


if __name__ == "__main__":
    main()
