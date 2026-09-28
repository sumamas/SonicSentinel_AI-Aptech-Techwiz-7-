from __future__ import annotations

import argparse
import hashlib
import uuid
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import librosa
import pandas as pd
import soundfile as sf

from src.ml.training_config import RAW_DIR, FOLDER_TO_LABEL, ALLOWED_AUDIO_EXTENSIONS


METADATA_COLUMNS = [
    "audio_id",
    "filename",
    "relative_path",
    "class_label",
    "duration",
    "sample_rate",
    "channels",
    "file_size_bytes",
    "sha256",
    "duplicate_of",
    "source",
    "environment",
    "device",
    "distance",
    "original_or_augmented",
    "split",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_info(path: Path):
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception:
        # MP3/M4A support can depend on local codecs. librosa/audioread gives a
        # useful fallback for duration/sample rate when SoundFile cannot decode.
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if getattr(y, "ndim", 1) == 1:
            channels = 1
            samples = len(y)
        else:
            channels = int(y.shape[0])
            samples = int(y.shape[-1])
        return float(samples / sr), int(sr), channels


def stable_audio_id(relative_path: str) -> str:
    value = uuid.uuid5(uuid.NAMESPACE_URL, f"sonicsentinel:{relative_path}")
    return f"AUD-{value.hex[:16].upper()}"


def build_metadata() -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    failures: list[dict] = []
    first_hash_owner: dict[str, str] = {}

    for folder, label in FOLDER_TO_LABEL.items():
        folder_path = RAW_DIR / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        for path in sorted(folder_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
                continue

            relative = path.relative_to(ROOT).as_posix()
            audio_id = stable_audio_id(relative)
            try:
                duration, sample_rate, channels = audio_info(path)
                file_hash = sha256_file(path)
                duplicate_of = first_hash_owner.get(file_hash, "")
                if not duplicate_of:
                    first_hash_owner[file_hash] = audio_id

                rows.append({
                    "audio_id": audio_id,
                    "filename": path.name,
                    "relative_path": relative,
                    "class_label": label,
                    "duration": round(duration, 4),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "file_size_bytes": path.stat().st_size,
                    "sha256": file_hash,
                    "duplicate_of": duplicate_of,
                    "source": "",
                    "environment": "",
                    "device": "",
                    "distance": "",
                    "original_or_augmented": "original",
                    "split": "",
                })
            except Exception as exc:
                failures.append({"relative_path": relative, "error": str(exc)})

    return pd.DataFrame(rows, columns=METADATA_COLUMNS), failures


def preserve_manual_metadata(new_df: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    if not old_path.exists() or new_df.empty:
        return new_df
    try:
        old_df = pd.read_csv(old_path).fillna("")
    except Exception:
        return new_df

    manual_columns = ["source", "environment", "device", "distance", "split"]
    if "relative_path" not in old_df.columns:
        return new_df

    old_df = old_df.set_index("relative_path")
    for index, row in new_df.iterrows():
        relative = row["relative_path"]
        if relative not in old_df.index:
            continue
        for column in manual_columns:
            if column in old_df.columns:
                value = old_df.loc[relative, column]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                if value != "" and not pd.isna(value):
                    new_df.at[index, column] = value
    return new_df


def main():
    parser = argparse.ArgumentParser(description="Build SonicSentinel dataset metadata.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "metadata.csv"),
        help="Metadata CSV output path.",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df, failures = build_metadata()
    df = preserve_manual_metadata(df, out)
    df.to_csv(out, index=False)

    failure_path = ROOT / "reports" / "metadata_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["relative_path", "error"]).to_csv(failure_path, index=False)

    print(f"Metadata records: {len(df)}")
    print(f"Decode failures:   {len(failures)}")
    print(f"Exact duplicates:  {(df['duplicate_of'].astype(str) != '').sum() if not df.empty else 0}")
    print(f"Saved: {out}")
    if failures:
        print(f"Failure report: {failure_path}")


if __name__ == "__main__":
    main()
