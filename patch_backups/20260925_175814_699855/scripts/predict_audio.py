from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.predict import predict_file


def main():
    parser = argparse.ArgumentParser(description="Predict one audio file with a trained SonicSentinel model.")
    parser.add_argument("audio")
    parser.add_argument(
        "--model",
        choices=["selected", "random_forest", "svm", "custom_cnn"],
        default="selected",
    )
    args = parser.parse_args()

    path = Path(args.audio)
    if not path.exists():
        raise SystemExit(f"Audio file not found: {path}")
    result = predict_file(str(path), args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
