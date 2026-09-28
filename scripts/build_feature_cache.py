from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.dataset import load_feature_dataset, save_feature_cache
from src.ml.training_config import FEATURE_DIR, SEED


def main():
    parser = argparse.ArgumentParser(description="Extract and cache features for RF/SVM training.")
    parser.add_argument(
        "--augment-train",
        type=int,
        default=1,
        help="Number of in-memory augmented feature variants per training original. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split in ("train", "validation", "test"):
        augment = args.augment_train if split == "train" else 0
        print(f"Extracting {split} features (augment_per_sample={augment})...")
        X, y, ids, failed = load_feature_dataset(
            ROOT,
            split,
            augment_per_sample=augment,
            seed=args.seed,
        )
        out = save_feature_cache(split, X, y, ids)
        summary[split] = {
            "samples": int(len(y)),
            "feature_count": int(X.shape[1]),
            "failed": len(failed),
            "cache": str(out.relative_to(ROOT)),
        }
        print(f"  X={X.shape} y={y.shape} failed={len(failed)} -> {out}")

    (FEATURE_DIR / "feature_cache_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Feature cache complete.")


if __name__ == "__main__":
    main()
