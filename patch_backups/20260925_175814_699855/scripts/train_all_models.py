from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n" + "=" * 72)
    print("RUN:", " ".join(command))
    print("=" * 72)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser(description="Train all three SonicSentinel custom/baseline models.")
    parser.add_argument("--quick", action="store_true", help="Smaller RF/SVM tuning grids.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-test", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    rf = [py, "-m", "src.ml.train_rf"]
    svm = [py, "-m", "src.ml.train_svm"]
    if args.quick:
        rf.append("--quick")
        svm.append("--quick")

    run(rf)
    run(svm)
    run([
        py,
        "-m",
        "src.ml.train_cnn",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
    ])

    evaluation = [py, "scripts/evaluate_models.py", "--batch-size", str(args.batch_size)]
    if args.noise_test:
        evaluation.append("--noise")
    run(evaluation)
    run([py, "scripts/compare_models.py"])

    print("\nAll model training/evaluation steps completed.")


if __name__ == "__main__":
    main()
