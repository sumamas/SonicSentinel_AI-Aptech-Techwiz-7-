"""Create a compact local code/model review package, without raw audio or credentials."""
from __future__ import annotations
import json
import argparse
import platform
import sys
import zipfile
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(ROOT), help="Project root; useful before installing the patch.")
    args = parser.parse_args()
    ROOT = Path(args.project).resolve()
    if not (ROOT / "src/ml/training_config.py").is_file():
        raise SystemExit("Wrong project folder: src/ml/training_config.py missing.")
    sys.path.insert(0, str(ROOT))
    packages = {}
    for name in ("scikit-learn", "numpy", "scipy", "librosa", "joblib", "tensorflow", "keras", "pandas", "soundfile"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    audit = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "models": {}}
    try:
        from src.ml.training_config import CLASSES
        from src.ml.predict import _load_joblib_model, cnn_labels, model_digest, get_selected_model_name
        audit["active_classes"] = CLASSES
        for key in ("random_forest", "svm", "custom_cnn"):
            try:
                labels = cnn_labels() if key == "custom_cnn" else [str(x) for x in _load_joblib_model(key)["model"].classes_]
                audit["models"][key] = {"classes": labels, "sha256": model_digest(key)}
            except Exception as exc:
                audit["models"][key] = {"error": str(exc)}
        try:
            audit["selected_model"] = get_selected_model_name()
        except Exception as exc:
            audit["selection_error"] = str(exc)
    except Exception as exc:
        audit["import_error"] = str(exc)

    paths = []
    for folder in ("src", "scripts", "routes", "static/js", "reports/model_evaluation", "data/splits"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for name in ("models/random_forest.joblib", "models/svm.joblib", "models/custom_cnn.keras",
                 "models/custom_cnn_manifest.json", "models/selected_model.json", "config/model_policy.json",
                 "data/metadata.csv", "reports/dataset_validation.csv", "reports/dataset_validation_summary.json",
                 "reports/cat_comparison.json", "requirements.txt", "requirements-ml.txt"):
        p = ROOT / name
        if p.is_file():
            paths.append(p)
    destination = ROOT / "SonicSentinel_Model_Review.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("runtime_audit.json", json.dumps(audit, indent=2))
        for p in sorted(set(paths)):
            archive.write(p, p.relative_to(ROOT).as_posix())
    print(json.dumps(audit, indent=2))
    print("Review ZIP:", destination)
    print("Size:", round(destination.stat().st_size / 1024**2, 1), "MB")
    print("Contains code, trained models, dataset metadata and reports. Raw recordings, .env and database files are excluded.")


if __name__ == "__main__":
    main()
