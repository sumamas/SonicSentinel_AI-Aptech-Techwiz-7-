from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.training_config import SPLITS_DIR, SEED


def main():
    parser = argparse.ArgumentParser(description="Create leakage-safe 70/15/15 dataset splits.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    metadata_path = ROOT / "data" / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit("metadata.csv not found. Run scripts/create_metadata.py first.")

    df = pd.read_csv(metadata_path).fillna("")
    if df.empty:
        raise SystemExit("metadata.csv is empty. Add audio files first.")

    # Final dataset requires unique originals. Exact duplicates are kept in the
    # metadata audit trail but excluded from model splits to avoid leakage.
    if "duplicate_of" in df.columns:
        split_df = df[df["duplicate_of"].astype(str).str.strip() == ""].copy()
    else:
        split_df = df.drop_duplicates(subset=["sha256"] if "sha256" in df.columns else ["relative_path"]).copy()

    counts = split_df["class_label"].value_counts()
    too_small = counts[counts < 7]
    if not too_small.empty:
        raise SystemExit(
            "Each class needs at least 7 unique originals for a stable stratified 70/15/15 split. "
            f"Too small: {too_small.to_dict()}"
        )

    train_df, temp_df = train_test_split(
        split_df,
        test_size=0.30,
        stratify=split_df["class_label"],
        random_state=args.seed,
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class_label"],
        random_state=args.seed,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_map: dict[str, str] = {}
    for name, part in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        part = part.copy().sort_values(["class_label", "audio_id"])
        part["split"] = name
        part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        split_map.update({audio_id: name for audio_id in part["audio_id"]})

    df["split"] = df["audio_id"].map(split_map).fillna("")
    df.to_csv(metadata_path, index=False)

    print("Created stratified 70/15/15 split using unique original recordings only.\n")
    for name in ("train", "validation", "test"):
        part = pd.read_csv(SPLITS_DIR / f"{name}.csv")
        print(f"{name:10} {len(part):5} recordings")
        print(part["class_label"].value_counts().sort_index().to_string())
        print()


if __name__ == "__main__":
    main()
