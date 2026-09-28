from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import librosa

from src.ml.training_config import RAW_DIR, FOLDER_TO_LABEL, ALLOWED_AUDIO_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description="Validate SonicSentinel raw dataset.")
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=0.30)
    parser.add_argument("--max-duration", type=float, default=60.0)
    args = parser.parse_args()

    detail_rows: list[dict] = []
    class_counts: dict[str, int] = {}
    usable_counts: dict[str, int] = {}
    seen_hashes: dict[str, str] = {}

    metadata_path = ROOT / "data" / "metadata.csv"
    metadata = pd.read_csv(metadata_path).fillna("") if metadata_path.exists() else pd.DataFrame()
    hash_lookup = {}
    if not metadata.empty and "relative_path" in metadata.columns and "sha256" in metadata.columns:
        hash_lookup = dict(zip(metadata["relative_path"], metadata["sha256"]))

    for folder, label in FOLDER_TO_LABEL.items():
        path = RAW_DIR / folder
        path.mkdir(parents=True, exist_ok=True)
        audio_files = [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
        ]
        class_counts[label] = len(audio_files)
        usable = 0

        for audio_path in sorted(audio_files):
            relative = audio_path.relative_to(ROOT).as_posix()
            status = "OK"
            error = ""
            duration = 0.0
            sample_rate = 0
            peak = 0.0
            rms = 0.0

            try:
                y, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
                if y.size == 0:
                    raise ValueError("No audio samples")
                duration = float(len(y) / sample_rate)
                peak = float(np.max(np.abs(y)))
                rms = float(np.sqrt(np.mean(np.square(y))))

                if duration < args.min_duration:
                    status = "TOO_SHORT"
                elif duration > args.max_duration:
                    status = "TOO_LONG"
                elif rms < 1e-5:
                    status = "NEAR_SILENT"
                else:
                    usable += 1
            except Exception as exc:
                status = "DECODE_FAILED"
                error = str(exc)

            file_hash = str(hash_lookup.get(relative, ""))
            duplicate_of = ""
            if file_hash:
                if file_hash in seen_hashes:
                    duplicate_of = seen_hashes[file_hash]
                    status = "DUPLICATE"
                else:
                    seen_hashes[file_hash] = relative

            detail_rows.append({
                "relative_path": relative,
                "class_label": label,
                "status": status,
                "duration": round(duration, 4),
                "sample_rate": sample_rate,
                "peak": round(peak, 6),
                "rms": round(rms, 6),
                "duplicate_of": duplicate_of,
                "error": error,
            })

        usable_counts[label] = usable

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    detail_path = report_dir / "dataset_validation.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    missing_classes = [label for label, count in usable_counts.items() if count == 0]
    below_minimum = {
        label: count for label, count in usable_counts.items()
        if count < args.minimum_per_class
    }
    invalid_count = sum(row["status"] != "OK" for row in detail_rows)

    summary = {
        "raw_total": int(sum(class_counts.values())),
        "usable_total": int(sum(usable_counts.values())),
        "invalid_or_duplicate_total": int(invalid_count),
        "minimum_per_class_requested": args.minimum_per_class,
        "class_counts": class_counts,
        "usable_counts": usable_counts,
        "missing_classes": missing_classes,
        "classes_below_minimum": below_minimum,
        "ready_for_development": not missing_classes and not below_minimum,
        "ready_for_final_300_per_class": all(count >= 300 for count in usable_counts.values()),
    }

    summary_path = report_dir / "dataset_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSonicSentinel dataset validation")
    print("=" * 38)
    for label in FOLDER_TO_LABEL.values():
        print(f"{label:28} {usable_counts[label]:4} usable / {class_counts[label]:4} files")
    print("-" * 38)
    print(f"Usable total: {summary['usable_total']}")
    print(f"Invalid/duplicate: {summary['invalid_or_duplicate_total']}")
    print(f"Development minimum ({args.minimum_per_class}/class): {'PASS' if summary['ready_for_development'] else 'NOT YET'}")
    print(f"Final 300/class target: {'PASS' if summary['ready_for_final_300_per_class'] else 'NOT YET'}")
    print(f"Detail report: {detail_path}")

    if missing_classes:
        raise SystemExit("Validation failed: one or more mandatory classes contain no usable audio.")


if __name__ == "__main__":
    main()
