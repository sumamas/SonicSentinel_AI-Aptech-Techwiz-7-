from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio.decode import decode_mono
import numpy as np
import pandas as pd
from src.ml.training_config import RAW_DIR, FOLDER_TO_LABEL, ALLOWED_AUDIO_EXTENSIONS
from src.ml.reliability import sha256_file, SRS_FOLDERS


def audit_dataset(root, raw_dir, folders, extensions, min_duration=.30, max_duration=60):
    rows = []
    hash_groups = defaultdict(list)
    for folder, label in folders.items():
        for path in sorted((raw_dir / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            row = {"relative_path": path.relative_to(root).as_posix(), "class_label": label,
                   "status": "OK", "duration": 0., "sample_rate": 0, "peak": 0., "rms": 0.,
                   "sha256": "", "duplicate_of": "", "error": ""}
            try:
                row["sha256"] = sha256_file(path)  # independent of any old metadata CSV
                y, sr = decode_mono(path)
                if not y.size or not np.all(np.isfinite(y)):
                    raise ValueError("Empty or non-finite audio")
                row.update(duration=float(len(y) / sr), sample_rate=int(sr),
                           peak=float(np.max(np.abs(y))), rms=float(np.sqrt(np.mean(y ** 2))))
                if row["duration"] < min_duration:
                    row["status"] = "TOO_SHORT"
                elif row["duration"] > max_duration:
                    row["status"] = "TOO_LONG"
                elif row["rms"] < 1e-5:
                    row["status"] = "NEAR_SILENT"
            except Exception as exc:
                row["status"], row["error"] = "DECODE_FAILED", str(exc)
            rows.append(row)
            if row["sha256"]:
                hash_groups[row["sha256"]].append(row)
    for group in hash_groups.values():
        if len({row["class_label"] for row in group}) > 1:
            for row in group:
                row["status"] = "LABEL_CONFLICT"
                row["error"] = "Identical bytes have different labels; review ALL copies."
        elif len(group) > 1:
            owner = group[0]["relative_path"]
            for row in group[1:]:
                row["status"], row["duplicate_of"] = "DUPLICATE", owner
    columns = ["relative_path", "class_label", "status", "duration", "sample_rate", "peak", "rms", "sha256", "duplicate_of", "error"]
    return pd.DataFrame(rows, columns=columns)


def summarize(report, folders, minimum):
    raw = report.class_label.value_counts().to_dict()
    usable = report.loc[report.status.eq("OK"), "class_label"].value_counts().to_dict()
    counts = {label: int(raw.get(label, 0)) for label in folders.values()}
    good = {label: int(usable.get(label, 0)) for label in folders.values()}
    return {
        "raw_total": len(report), "usable_total": sum(good.values()),
        "invalid_or_duplicate_total": int(report.status.ne("OK").sum()),
        "class_counts": counts, "usable_counts": good,
        "minimum_per_class_requested": minimum,
        "missing_classes": [label for label, n in good.items() if n == 0],
        "classes_below_minimum": {label: n for label, n in good.items() if n < minimum},
        "ready_for_development": all(n >= minimum for n in good.values()),
        "raw_unique_file_300_per_class_check": all(usable.get(label, 0) >= 300 for label in SRS_FOLDERS.values()),
        "note": "File counts alone cannot prove unique original sources. Edited/transcoded copies require original_audio_id review.",
    }


def main():
    parser = argparse.ArgumentParser(description="Audit raw audio; count usable files after duplicate/conflict checks.")
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=.30)
    parser.add_argument("--max-duration", type=float, default=60.)
    parser.add_argument("--require-all-classes", action="store_true")
    args = parser.parse_args()
    folders = SRS_FOLDERS if args.require_all_classes else FOLDER_TO_LABEL
    report = audit_dataset(ROOT, RAW_DIR, folders, ALLOWED_AUDIO_EXTENSIONS, args.min_duration, args.max_duration)
    summary = summarize(report, folders, args.minimum_per_class)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    report.to_csv(out / "dataset_validation.csv", index=False)
    (out / "dataset_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SonicSentinel dataset validation (unique usable FILES)")
    for label in folders.values():
        print(f"{label:28} {summary['usable_counts'][label]:4} usable / {summary['class_counts'][label]:4} files")
    print("Usable:", summary["usable_total"], "Invalid/duplicate:", summary["invalid_or_duplicate_total"], "Raw:", summary["raw_total"])
    print("Development minimum:", "PASS" if summary["ready_for_development"] else "NOT YET")
    print("All 10 classes, 300 unique files/class:", "PASS" if summary["raw_unique_file_300_per_class_check"] else "NOT YET")
    print("Original-source independence needs human review; preparation reports and verify_prepared.py check frozen split integrity.")
    if not summary["ready_for_development"]:
        raise SystemExit("Validation incomplete: add/review audio in the classes below the requested minimum.")


if __name__ == "__main__":
    main()
