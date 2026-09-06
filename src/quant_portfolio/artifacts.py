"""Aligned, fingerprinted local artifacts for the independent synthetic demo."""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .blend import weighted_directional_accuracy
from .data import Panel, panel_fingerprint
from .models import MODEL_NAMES
from .provenance import (
    fold_protocol, json_bytes, shared_protocol, validate_prediction_context,
)


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
    """Publish a complete generation; interrupted writes retain the previous run.

    Prediction metadata is attached by model_oof/refit_predict. It records the
    actual call settings; manually supplied or modified predictions are rejected.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model artifact name.")
    panel.validate()
    check_prediction(oof, panel.train.index)
    check_prediction(test, panel.test.index)
    oof_record = validate_prediction_context(oof, "oof")
    test_record = validate_prediction_context(test, "full_refit")
    context = oof_record["context"]
    if context != test_record["context"]:
        raise ValueError("OOF and full-refit generation contexts do not match.")
    if context["panel_fingerprint"] != panel_fingerprint(panel) or context["model"] != model_name:
        raise ValueError("Prediction context does not match the model/panel being saved.")
    if params is not None and dict(params) != context["params"]:
        raise ValueError("Saved parameters must match prediction generation parameters.")
    protocol = oof_record["fold_protocol"]
    if protocol != fold_protocol(panel, protocol["n_splits"]):
        raise ValueError("OOF fold assignments do not match the panel.")
    payloads = {}
    for name, values in (("oof", oof), ("test_raw", test)):
        payload = values.rename("prediction").to_csv(index_label="row_id").encode("utf-8")
        restored = _read_prediction(payload, values.index)
        check_prediction(restored, values.index)
        np.testing.assert_allclose(restored.to_numpy(), values.to_numpy(), rtol=1e-12, atol=1e-14)
        payloads[name + ".csv"] = payload
    manifest = {
        "schema_version": 2,
        "data_kind": "independent_synthetic_demo",
        "model": model_name,
        "panel_fingerprint": panel_fingerprint(panel),
        "params": context["params"],
        "generation_context": context,
        "fold_protocol": protocol,
        "oof_selection_score": weighted_directional_accuracy(panel.y, oof),
    }
    return commit_bundle(Path(output_dir) / model_name, manifest, payloads)


def _replace_manifest(destination: Path, payload: bytes) -> None:
    """One atomic pointer switch after every immutable output has been written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".manifest-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination / "manifest.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def commit_bundle(destination: Path, metadata: dict, payloads: dict[str, bytes]) -> dict:
    """Keep prior generations readable; orphan runs are never active manifests.

    A single writer per model/output directory is required. This is an atomic
    publication boundary for ordinary process interruptions, not a power-loss
    durability guarantee across every filesystem.
    """
    for filename in payloads:
        if not re.fullmatch(r"[a-z_]+\.(csv|json)", filename):
            raise ValueError("Bundle filenames must be simple CSV/JSON names.")
    run_id = uuid4().hex
    generation = destination / "runs" / run_id
    manifest = {**metadata, "run_id": run_id, "files": {
        filename: {"filename": f"runs/{run_id}/{filename}", "sha256": hashlib.sha256(payload).hexdigest()}
        for filename, payload in payloads.items()
    }}
    manifest_payload = json_bytes(manifest)  # Validate JSON before touching disk.
    generation.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        with (generation / filename).open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    _replace_manifest(destination, manifest_payload)
    return manifest


def _read_bundle(destination: Path, expected_files: set[str], snapshot: dict | None = None):
    raw = (destination / "manifest.json").read_bytes() if snapshot is None else json_bytes(snapshot)
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 2:
        raise ValueError("Artifact schema is outdated; regenerate all three models in a fresh output directory.")
    run_id = manifest.get("run_id", "")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid artifact generation identifier.")
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("The artifact generation has an unexpected file set.")
    payloads = {}
    for filename in sorted(expected_files):
        record = manifest["files"][filename]
        expected = f"runs/{run_id}/{filename}"
        if record["filename"] != expected:
            raise ValueError("Unexpected artifact filename; arbitrary manifest paths are not accepted.")
        path = destination / expected
        if path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Artifact path leaves its generation directory.")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Artifact file hash mismatch; a prediction or report file has changed.")
        payloads[filename] = payload
    return manifest, payloads, hashlib.sha256(raw).hexdigest()


def _read_prediction(payload: bytes, expected_index: pd.Index) -> pd.Series:
    # Preserve string IDs such as '001' rather than letting CSV inference turn them into 1.
    dtype = {"row_id": str} if pd.api.types.is_string_dtype(expected_index.dtype) else None
    table = pd.read_csv(BytesIO(payload), index_col="row_id", dtype=dtype)
    if list(table.columns) != ["prediction"]:
        raise ValueError("An artifact must contain exactly one prediction column.")
    values = table.prediction
    check_prediction(values, expected_index)
    return values


def load_prediction_stack(output_dir: str | Path, panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require the same panel, fold assignments, seed, provider and runtime.

    Different model families and parameters are expected. Mixing different fold
    protocols can be a valid separate experiment, but is not this public workflow.
    """
    panel.validate()
    expected_fingerprint = panel_fingerprint(panel)
    frames: dict[str, dict[str, pd.Series]] = {"oof": {}, "test_raw": {}}
    sources = {}
    snapshots = {}
    common_protocol = None
    for model_name in MODEL_NAMES:
        destination = Path(output_dir) / model_name
        try:
            manifest, payloads, manifest_hash = _read_bundle(destination, {"oof.csv", "test_raw.csv"})
        except FileNotFoundError as exc:
            raise ValueError(f"Missing {model_name} artifacts; run all three model notebooks or --model all first.") from exc
        if manifest.get("model") != model_name:
            raise ValueError("Unknown or inconsistent artifact schema/model.")
        if manifest.get("data_kind") != "independent_synthetic_demo":
            raise ValueError("The public example only loads its own synthetic artifacts.")
        if manifest.get("panel_fingerprint") != expected_fingerprint:
            raise ValueError("Artifact data fingerprint mismatch; regenerate all models on the same panel.")
        context = manifest["generation_context"]
        if context["model"] != model_name or context["panel_fingerprint"] != expected_fingerprint:
            raise ValueError("Generation context does not match its model/panel.")
        if context["params"] != manifest["params"]:
            raise ValueError("Artifact parameters and generation context do not match.")
        protocol = manifest["fold_protocol"]
        if protocol != fold_protocol(panel, protocol["n_splits"]):
            raise ValueError("Artifact fold assignments do not match the panel.")
        shared = shared_protocol(context, protocol)
        if common_protocol is not None and shared != common_protocol:
            raise ValueError("Model experiment protocols differ; regenerate with shared folds, seed, provider and runtime.")
        common_protocol = shared
        sources[model_name] = manifest_hash
        snapshots[model_name] = manifest
        for kind, expected_index in (("oof", panel.train.index), ("test_raw", panel.test.index)):
            frames[kind][model_name] = _read_prediction(payloads[kind + ".csv"], expected_index)
    oof, test = pd.DataFrame(frames["oof"]), pd.DataFrame(frames["test_raw"])
    for frame in (oof, test):
        frame.attrs["source_manifests"] = sources.copy()
        frame.attrs["source_records"] = snapshots.copy()
        frame.attrs["shared_protocol"] = common_protocol
    return oof, test


ENSEMBLE_FILES = {
    "synthetic_oof_raw.csv", "synthetic_test_raw.csv", "synthetic_submission.csv",
    "synthetic_weight_candidates.csv", "synthetic_weight_diagnostics.csv", "synthetic_summary.json",
}


def verify_ensemble_artifacts(output_dir: str | Path, panel: Panel) -> dict:
    """Verify the saved generation and its pinned inputs, including after a rerun.

    Old model generations are retained, so a newer active model manifest does
    not invalidate a correctly pinned earlier ensemble. Hashes detect accidental
    changes; these local records are not signatures or leaderboard evidence.
    """
    panel.validate()
    root = Path(output_dir)
    manifest, payloads, _ = _read_bundle(root / "ensemble", ENSEMBLE_FILES)
    if manifest.get("model") != "ensemble" or manifest.get("panel_fingerprint") != panel_fingerprint(panel):
        raise ValueError("Ensemble data/model context mismatch.")
    sources = manifest.get("sources", {})
    if set(sources) != set(MODEL_NAMES):
        raise ValueError("Ensemble must identify all three source models.")
    source_predictions = {"oof.csv": {}, "test_raw.csv": {}}
    for model in MODEL_NAMES:
        record = sources[model]
        snapshot, inputs, digest = _read_bundle(root / model, {"oof.csv", "test_raw.csv"}, record["manifest"])
        if digest != record["manifest_sha256"] or snapshot["model"] != model:
            raise ValueError("Ensemble source manifest hash/model mismatch.")
        if shared_protocol(snapshot["generation_context"], snapshot["fold_protocol"]) != manifest["shared_protocol"]:
            raise ValueError("Ensemble source experiment protocol mismatch.")
        for name, index in (("oof.csv", panel.train.index), ("test_raw.csv", panel.test.index)):
            source_predictions[name][model] = _read_prediction(inputs[name], index)
    summary = json.loads(payloads["synthetic_summary.json"])
    if summary != manifest["selection_summary"]:
        raise ValueError("Ensemble selection summary changed or is inconsistent.")
    if summary["weights"] != manifest["weights"] or summary["selection_config"] != manifest["selection_config"]:
        raise ValueError("Ensemble summary does not match its manifest.")
    weights = pd.Series(manifest["weights"]).reindex(MODEL_NAMES)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid ensemble weights.")
    for source_name, output_name, index in (
        ("oof.csv", "synthetic_oof_raw.csv", panel.train.index),
        ("test_raw.csv", "synthetic_test_raw.csv", panel.test.index),
    ):
        expected = pd.DataFrame(source_predictions[source_name]) @ weights
        actual = _read_prediction(payloads[output_name], index)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    signs = _read_prediction(payloads["synthetic_submission.csv"], panel.test.index)
    raw_test = _read_prediction(payloads["synthetic_test_raw.csv"], panel.test.index)
    np.testing.assert_array_equal(signs, np.where(raw_test >= 0, 1, -1))
    return manifest
