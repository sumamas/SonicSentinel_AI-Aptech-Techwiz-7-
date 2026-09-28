from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.audio_prepare import (  # noqa: E402
    prepare_audio,
    quality_metrics,
    read_audio_info,
    sha256_file,
    write_wav,
)
from src.dataset.starter_config import (  # noqa: E402
    ACTIVE_CLASSES,
    RANDOM_SEED,
    SPLIT_RATIOS,
    STARTER_MAX_ORIGINAL_DURATION_SECONDS,
    SUPPORTED_EXTENSIONS,
    TARGET_DURATION_SECONDS,
    TARGET_SAMPLE_RATE,
)


@dataclass
class AudioRow:
    audio_id: str
    class_folder: str
    class_label: str
    original_filename: str
    original_path: str
    extension: str
    original_duration_seconds: float | None
    original_sample_rate: int | None
    original_channels: int | None
    sha256: str | None
    duplicate_of: str | None
    status: str
    exclusion_reason: str | None
    split: str | None
    prepared_path: str | None
    prepared_duration_seconds: float | None
    prepared_sample_rate: int | None
    rms: float | None
    silence_ratio: float | None
    clipping_ratio: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the SonicSentinel 3-class starter dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Raw dataset root containing class folders (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "starter_3class",
        help="Prepared dataset output directory",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "starter_3class",
        help="Dataset report directory",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete old prepared output before rebuilding it",
    )
    return parser.parse_args()


def discover(raw_dir: Path) -> list[AudioRow]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw_dir}")

    rows: list[AudioRow] = []
    counters: dict[str, int] = defaultdict(int)
    prefix = {
        "alarm_siren": "AS",
        "glass_breaking": "GB",
        "gunshot": "GS",
    }

    for class_folder, class_label in ACTIVE_CLASSES.items():
        folder = raw_dir / class_folder
        if not folder.exists():
            print(f"WARNING: Missing class folder: {folder}")
            continue

        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            counters[class_folder] += 1
            audio_id = f"{prefix[class_folder]}_{counters[class_folder]:04d}"

            try:
                duration, sr, channels = read_audio_info(path)
                digest = sha256_file(path)
                status = "candidate"
                reason = None
            except Exception as exc:
                duration = sr = channels = None
                digest = None
                status = "excluded_decode_error"
                reason = str(exc)

            rows.append(
                AudioRow(
                    audio_id=audio_id,
                    class_folder=class_folder,
                    class_label=class_label,
                    original_filename=path.name,
                    original_path=str(path.relative_to(PROJECT_ROOT))
                    if path.is_relative_to(PROJECT_ROOT)
                    else str(path),
                    extension=path.suffix.lower(),
                    original_duration_seconds=round(duration, 6) if duration is not None else None,
                    original_sample_rate=sr,
                    original_channels=channels,
                    sha256=digest,
                    duplicate_of=None,
                    status=status,
                    exclusion_reason=reason,
                    split=None,
                    prepared_path=None,
                    prepared_duration_seconds=None,
                    prepared_sample_rate=None,
                    rms=None,
                    silence_ratio=None,
                    clipping_ratio=None,
                )
            )
    return rows


def apply_exact_duplicate_filter(rows: list[AudioRow]) -> None:
    # Keep first lexicographic candidate for a hash; do not delete the source file.
    seen: dict[str, AudioRow] = {}
    for row in rows:
        if row.status != "candidate" or not row.sha256:
            continue
        if row.sha256 in seen:
            row.status = "excluded_exact_duplicate"
            row.duplicate_of = seen[row.sha256].audio_id
            row.exclusion_reason = f"Exact SHA-256 duplicate of {row.duplicate_of}"
        else:
            seen[row.sha256] = row


def apply_starter_outlier_rules(rows: list[AudioRow]) -> None:
    for row in rows:
        if row.status != "candidate":
            continue
        if row.original_duration_seconds is None:
            continue
        if row.original_duration_seconds > STARTER_MAX_ORIGINAL_DURATION_SECONDS:
            row.status = "excluded_long_outlier"
            row.exclusion_reason = (
                f"Starter pipeline excludes originals longer than "
                f"{STARTER_MAX_ORIGINAL_DURATION_SECONDS:g}s to prevent one recording "
                "from dominating the small prototype. Segment it later with group-aware splitting."
            )


def allocate_splits(rows: list[AudioRow]) -> None:
    rng = random.Random(RANDOM_SEED)
    by_class: dict[str, list[AudioRow]] = defaultdict(list)
    for row in rows:
        if row.status == "candidate":
            by_class[row.class_folder].append(row)

    for class_folder, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            raise RuntimeError(
                f"Class {class_folder} has only {n} usable samples; at least 3 are needed."
            )

        # Deterministic class-wise 70/15/15 split with at least one val/test item.
        n_test = max(1, int(round(n * SPLIT_RATIOS["test"])))
        n_val = max(1, int(round(n * SPLIT_RATIOS["validation"])))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
        n_train = n - n_val - n_test

        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]

        for row in train_items:
            row.split = "train"
        for row in val_items:
            row.split = "validation"
        for row in test_items:
            row.split = "test"


def process_audio(rows: list[AudioRow], output_dir: Path, raw_dir: Path) -> None:
    # Map raw paths safely even when the uploaded ZIP was extracted outside project root.
    for row in rows:
        if row.status != "candidate" or not row.split:
            continue
        source = raw_dir / row.class_folder / row.original_filename
        try:
            y = prepare_audio(source)
            metrics = quality_metrics(y)
            out_path = output_dir / row.split / row.class_folder / f"{row.audio_id}.wav"
            write_wav(out_path, y)

            row.status = "included"
            row.prepared_path = str(out_path.relative_to(PROJECT_ROOT)) \
                if out_path.is_relative_to(PROJECT_ROOT) else str(out_path)
            row.prepared_duration_seconds = TARGET_DURATION_SECONDS
            row.prepared_sample_rate = TARGET_SAMPLE_RATE
            row.rms = metrics["rms"]
            row.silence_ratio = metrics["silence_ratio"]
            row.clipping_ratio = metrics["clipping_ratio"]
        except Exception as exc:
            row.status = "excluded_preprocessing_error"
            row.exclusion_reason = str(exc)
            row.split = None


def dataframe(rows: list[AudioRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def write_reports(df: pd.DataFrame, output_dir: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_csv = output_dir / "metadata_full.csv"
    included_csv = output_dir / "metadata_included.csv"
    excluded_csv = output_dir / "metadata_excluded.csv"

    df.to_csv(full_csv, index=False)
    df[df["status"] == "included"].to_csv(included_csv, index=False)
    df[df["status"] != "included"].to_csv(excluded_csv, index=False)

    included = df[df["status"] == "included"]
    raw_candidate_count = int(len(df))
    report = {
        "mode": "starter_3class_prototype",
        "active_classes": ACTIVE_CLASSES,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "max_original_duration_seconds": STARTER_MAX_ORIGINAL_DURATION_SECONDS,
        "random_seed": RANDOM_SEED,
        "raw_audio_files": raw_candidate_count,
        "included_after_filters": int(len(included)),
        "excluded": int(raw_candidate_count - len(included)),
        "raw_counts_by_class": df.groupby("class_folder").size().astype(int).to_dict(),
        "included_counts_by_class": included.groupby("class_folder").size().astype(int).to_dict(),
        "split_counts": included.groupby(["class_folder", "split"]).size().astype(int).to_dict(),
        "status_counts": df["status"].value_counts().astype(int).to_dict(),
    }
    # JSON cannot have tuple keys.
    report["split_counts"] = {
        f"{class_folder}/{split}": count
        for (class_folder, split), count in report["split_counts"].items()
    }

    (reports_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "SonicSentinel Starter 3-Class Dataset Report",
        "=" * 47,
        "",
        f"Raw supported audio files: {report['raw_audio_files']}",
        f"Included after filters:    {report['included_after_filters']}",
        f"Excluded:                  {report['excluded']}",
        "",
        "Raw class counts:",
    ]
    for cls, count in report["raw_counts_by_class"].items():
        lines.append(f"  {cls}: {count}")
    lines.extend(["", "Included class/split counts:"])
    for cls in ACTIVE_CLASSES:
        class_rows = included[included["class_folder"] == cls]
        lines.append(
            f"  {cls}: total={len(class_rows)}, "
            f"train={int((class_rows['split'] == 'train').sum())}, "
            f"validation={int((class_rows['split'] == 'validation').sum())}, "
            f"test={int((class_rows['split'] == 'test').sum())}"
        )
    lines.extend(["", "Exclusion/status counts:"])
    for status, count in report["status_counts"].items():
        lines.append(f"  {status}: {count}")

    (reports_dir / "dataset_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    reports_dir = args.reports_dir.resolve()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.clean and reports_dir.exists():
        shutil.rmtree(reports_dir)

    print("SonicSentinel — Starter 3-Class Dataset Preparation")
    print(f"Raw folder: {raw_dir}")
    print(f"Output:     {output_dir}")
    print()

    rows = discover(raw_dir)
    print(f"Discovered {len(rows)} supported audio files.")

    apply_exact_duplicate_filter(rows)
    apply_starter_outlier_rules(rows)
    allocate_splits(rows)
    process_audio(rows, output_dir, raw_dir)

    df = dataframe(rows)
    write_reports(df, output_dir, reports_dir)

    included = df[df["status"] == "included"]
    excluded = df[df["status"] != "included"]
    print()
    print("Completed.")
    print(f"Included: {len(included)}")
    print(f"Excluded: {len(excluded)}")
    print()
    for cls in ACTIVE_CLASSES:
        part = included[included["class_folder"] == cls]
        print(
            f"{cls:18} total={len(part):2d}  "
            f"train={(part['split'] == 'train').sum():2d}  "
            f"val={(part['split'] == 'validation').sum():2d}  "
            f"test={(part['split'] == 'test').sum():2d}"
        )
    print()
    print(f"Metadata: {output_dir / 'metadata_full.csv'}")
    print(f"Report:   {reports_dir / 'dataset_report.txt'}")


if __name__ == "__main__":
    main()
