"""Aligned, fingerprinted local artifacts for the independent synthetic demo."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .blend import weighted_directional_accuracy
from .data import Panel, panel_fingerprint
from .models import MODEL_NAMES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_prediction(values: pd.Series, expected_index: pd.Index) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError("Predictions must be an indexed Series.")
    if not values.index.is_unique or not values.index.equals(expected_index):
        raise ValueError("Prediction IDs must exactly match the expected ordered IDs.")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Every prediction must be finite.")


def save_model_artifacts(
    output_dir: str | Path,
    model_name: str,
    panel: Panel,
    oof: pd.Series,
    test: pd.Series,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write only generated demo results, preserving ID order and raw scale.

    The manifest is written last, so a partial write cannot look like a valid
    new artifact set. Normal reruns replace that model's local demo artifacts.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model artifact name.")
    panel.validate()
    check_prediction(oof, panel.train.index)
    check_prediction(test, panel.test.index)
    destination = Path(output_dir) / model_name
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in (("oof", oof), ("test_raw", test)):
        path = destination / f"{name}.csv"
        values.rename("prediction").to_csv(path, index_label="row_id")
        restored = pd.read_csv(path, index_col="row_id")["prediction"]
        check_prediction(restored, values.index)
        np.testing.assert_allclose(restored.to_numpy(), values.to_numpy(), rtol=1e-12, atol=1e-14)
        files[name] = {"filename": path.name, "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 1,
        "data_kind": "independent_synthetic_demo",
        "model": model_name,
        "panel_fingerprint": panel_fingerprint(panel),
        "params": dict(params or {}),
        "oof_selection_score": weighted_directional_accuracy(panel.y, oof),
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_prediction_stack(output_dir: str | Path, panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject stale datasets, changed files, duplicate IDs, and positional blends."""
    panel.validate()
    expected_fingerprint = panel_fingerprint(panel)
    frames: dict[str, dict[str, pd.Series]] = {"oof": {}, "test_raw": {}}
    for model_name in MODEL_NAMES:
        destination = Path(output_dir) / model_name
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("model") != model_name:
            raise ValueError("Unknown or inconsistent artifact schema/model.")
        if manifest.get("data_kind") != "independent_synthetic_demo":
            raise ValueError("The public example only loads its own synthetic artifacts.")
        if manifest.get("panel_fingerprint") != expected_fingerprint:
            raise ValueError("Artifact data fingerprint mismatch; regenerate all models on the same panel.")
        for kind, expected_index in (("oof", panel.train.index), ("test_raw", panel.test.index)):
            record = manifest["files"][kind]
            if record["filename"] != f"{kind}.csv":
                raise ValueError("Unexpected artifact filename; arbitrary manifest paths are not accepted.")
            path = destination / record["filename"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("Artifact file hash mismatch; a prediction file has changed.")
            table = pd.read_csv(path, index_col="row_id")
            if list(table.columns) != ["prediction"]:
                raise ValueError("An artifact must contain exactly one prediction column.")
            values = table.prediction
            check_prediction(values, expected_index)
            frames[kind][model_name] = values
    return pd.DataFrame(frames["oof"]), pd.DataFrame(frames["test_raw"])

