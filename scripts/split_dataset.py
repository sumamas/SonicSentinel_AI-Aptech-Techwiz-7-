from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml.training_config import SPLITS_DIR, SEED, CLASSES
from src.ml.reliability import SRS_FOLDERS, sha256_file


def group_split(df, seed=42, minimum=20):
    df = df.copy()
    if "original_or_augmented" in df:
        df = df[df.original_or_augmented.str.lower().eq("original")].copy()
    if "original_audio_id" not in df:
        df["original_audio_id"] = df["audio_id"]
    df["original_audio_id"] = df["original_audio_id"].fillna("").astype(str).str.strip()
    blank = df.original_audio_id.eq("")
    df.loc[blank, "original_audio_id"] = df.loc[blank, "audio_id"]
    if df.audio_id.duplicated().any() or df.relative_path.duplicated().any():
        raise ValueError("Duplicate recording IDs or file paths in metadata.")
    conflicts = df.groupby("original_audio_id").class_label.nunique()
    if (conflicts > 1).any():
        raise ValueError("One original_audio_id has multiple labels; review the source labels first.")
    groups = df[["original_audio_id", "class_label"]].drop_duplicates()
    counts = groups.class_label.value_counts()
    if (counts < minimum).any():
        raise ValueError(f"Need at least {minimum} original source groups per class: {counts.to_dict()}")
    training, held = train_test_split(groups, test_size=.30, stratify=groups.class_label, random_state=seed)
    validation, test = train_test_split(held, test_size=.50, stratify=held.class_label, random_state=seed)
    return {name: df[df.original_audio_id.isin(part.original_audio_id)].copy()
            for name, part in (("train", training), ("validation", validation), ("test", test))}


def main():
    parser = argparse.ArgumentParser(description="Split audited originals by original_audio_id, then augment training only.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--require-all-classes", action="store_true")
    args = parser.parse_args()
    metadata_path = ROOT / "data" / "metadata.csv"
    report_path = ROOT / "reports" / "dataset_validation.csv"
    if not metadata_path.is_file() or not report_path.is_file():
        raise SystemExit("Run create_metadata.py, then validate_dataset.py first.")
    df = pd.read_csv(metadata_path).fillna("")
    report = pd.read_csv(report_path).fillna("")
    if "sha256" not in report or "sha256" not in df:
        raise SystemExit("Refresh metadata and the validation report with patched scripts.")
    good = report[report.status.eq("OK")][["relative_path", "sha256", "class_label"]]
    clean = df.merge(good, on=["relative_path", "sha256", "class_label"], how="inner", validate="one_to_one")
    wanted = set(SRS_FOLDERS.values()) if args.require_all_classes else set(CLASSES)
    if set(clean.class_label) != wanted or set(CLASSES) != wanted:
        raise SystemExit("Audited classes and active config disagree. Enable all 10 classes only in the NEW training workspace.")
    for _, row in clean.iterrows():
        path = ROOT / row.relative_path
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise SystemExit(f"Audio changed since validation: {row.relative_path}. Refresh metadata and validation.")
    parts = group_split(clean, args.seed, args.minimum_per_class)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_map = {}
    summary = {}
    for name, part in parts.items():
        part = part.sort_values(["class_label", "audio_id"])
        part["split"] = name
        part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        split_map.update(zip(part.audio_id, [name] * len(part)))
        summary[name] = {"recordings": len(part), "source_groups": part.original_audio_id.nunique(),
                         "per_class": part.class_label.value_counts().to_dict()}
        print(name, "recordings:", len(part), "source groups:", part.original_audio_id.nunique())
        print(part.class_label.value_counts().sort_index().to_string())
    df["split"] = df.audio_id.map(split_map).fillna("")
    df.to_csv(metadata_path, index=False)
    (SPLITS_DIR / "split_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("No source group crosses splits. Augment only AFTER splitting.")


if __name__ == "__main__":
    main()
