from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "starter_3class" / "metadata_included.csv"
OUT = ROOT / "data" / "splits"

if not META.exists():
    raise SystemExit(f"Missing {META}. Run prepare_starter_dataset.bat first.")

OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(META)

for split in ("train", "validation", "test"):
    part = df[df["split"] == split].copy()
    output = pd.DataFrame({
        "audio_id": part["audio_id"].astype(str),
        "class_label": part["class_label"].astype(str),
        "relative_path": part["prepared_path"].astype(str),
        "source_original_path": part["original_path"].astype(str),
        "sha256": part["sha256"].astype(str),
    })
    path = OUT / f"{split}.csv"
    output.to_csv(path, index=False)
    print(f"{split:10} {len(output):3d} -> {path.relative_to(ROOT)}")

print("Starter split manifests synced successfully.")
