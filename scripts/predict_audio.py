from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ml.predict import predict_file, predict_models, compare_predictions


def main():
    parser = argparse.ArgumentParser(description="Predict audio and report uncertainty without changing model scores.")
    parser.add_argument("audio")
    parser.add_argument("--model", choices=["selected", "random_forest", "svm", "custom_cnn"], default="selected")
    parser.add_argument("--compare", action="store_true", help="Compare the three Python models; this is not a GTM comparison.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()
    try:
        result = compare_predictions(predict_models(args.audio)) if args.compare else predict_file(args.audio, args.model)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
