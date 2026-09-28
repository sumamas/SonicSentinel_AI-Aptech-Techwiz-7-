"""Make a separate ten-class training workspace; never overwrite live models."""
from __future__ import annotations
import argparse
import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ml.reliability import SRS_FOLDERS, CRITICAL_RECALL_CLASSES


def all_classes_config(text):
    replacements = {
        "CLASSES": list(SRS_FOLDERS.values()), "FOLDER_TO_LABEL": SRS_FOLDERS,
        "CRITICAL_CLASSES": CRITICAL_RECALL_CLASSES,
    }
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                value = replacements[name]
                if isinstance(value, set):
                    rendered = "{" + ", ".join(repr(x) for x in sorted(value)) + "}"
                else:
                    rendered = repr(value)
                edits.append((node.lineno - 1, node.end_lineno, f"{name} = {rendered}\n"))
    if len(edits) != len(replacements):
        raise ValueError("Unrecognized training config. No active project files were modified.")
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", help="A NEW folder outside the active project.")
    args = parser.parse_args()
    dest = Path(args.destination).resolve()
    if dest.exists() or dest.is_relative_to(ROOT):
        raise SystemExit("Choose a new, nonexistent folder outside this project.")
    config_text = all_classes_config((ROOT / "src/ml/training_config.py").read_text())
    if not (ROOT / "data/raw").is_dir():
        raise SystemExit("Missing data/raw.")
    dest.mkdir(parents=True)
    for name in ("src", "scripts", "config"):
        if (ROOT / name).is_dir():
            shutil.copytree(ROOT / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("requirements.txt", "requirements-ml.txt"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, dest / name)
    (dest / "src/ml/training_config.py").write_text(config_text, encoding="utf-8")
    (dest / "data").mkdir(exist_ok=True)
    print("Copying original audio into new workspace...", flush=True)
    shutil.copytree(ROOT / "data/raw", dest / "data/raw")
    if (ROOT / "data/metadata.csv").is_file():
        shutil.copy2(ROOT / "data/metadata.csv", dest / "data/metadata.csv")
    print("New ten-class workspace:", dest)
    print("Previous models, features and splits were not copied.")
    print("Next: create_metadata.py -> review original_audio_id -> validate_dataset.py --require-all-classes --minimum-per-class 300")
    print("Then split_dataset.py --require-all-classes --minimum-per-class 300, build_feature_cache.py and train_all_models.py.")


if __name__ == "__main__":
    main()
