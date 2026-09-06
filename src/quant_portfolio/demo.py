"""Small public research workflow. No downloads or competition submissions."""

import argparse
import json
import os
from pathlib import Path

# Keep the tiny demonstration economical and predictable on laptops.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .artifacts import check_prediction, commit_bundle, load_prediction_stack, save_model_artifacts
from .blend import learn_common_weights, selection_fingerprint, weighted_directional_accuracy
from .data import Panel, make_synthetic_panel
from .models import MODEL_NAMES, model_oof, refit_predict
from .search import ridge_grid_search, run_optuna_search
from .provenance import indexed_fingerprint, json_bytes


def run_model(model_name: str, panel: Panel, output_dir: str | Path, search: bool = False):
    """Select demo parameters, rebuild OOF, and refit on every training row."""
    destination = Path(output_dir)
    params = {}
    if search:
        if model_name == "ridge":
            selected = ridge_grid_search(panel)
        else:
            selected = run_optuna_search(model_name, panel, destination / f"{model_name}_demo.sqlite3")
        params = selected.params
    oof = model_oof(model_name, panel, params)
    test = refit_predict(model_name, panel, params)
    manifest = save_model_artifacts(destination, model_name, panel, oof, test, params)
    print(f"{model_name}: synthetic OOF selection score = {manifest['oof_selection_score']:.6f}")
    return manifest


def run_ensemble(
    panel: Panel,
    output_dir: str | Path,
    epsilon: float | None = None,
    *,
    step: float | None = None,
    repeats: int | None = None,
    n_splits: int | None = None,
    seed: int | None = None,
    selected=None,
):
    """Export an existing selection without searching again, or learn it once.

    A supplied selection must match the current OOF values, model generations
    and any explicitly supplied controls. Everything is validated before a new
    generation becomes active. The returned object is the selection exported.
    """
    oof, test = load_prediction_stack(output_dir, panel)
    supplied = {key: value for key, value in {
        "epsilon": epsilon, "step": step, "repeats": repeats,
        "n_splits": n_splits, "seed": seed,
    }.items() if value is not None}
    if selected is None:
        config = {"epsilon": 0.05, "step": 0.05, "repeats": 4, "n_splits": 5, "seed": 42} | supplied
        selected = learn_common_weights(panel.y, oof, panel.train.day_id, **config)
    elif any(selected.config.get(key) != value for key, value in supplied.items()):
        raise ValueError("Export controls differ from the existing selection; select again explicitly.")
    if selected.input_fingerprint != indexed_fingerprint(panel.y, oof, panel.train.day_id):
        raise ValueError("The selected weights belong to different OOF inputs; select again.")
    if selected.source_manifests != oof.attrs["source_manifests"]:
        raise ValueError("Model generations changed after weight selection; reload and select again.")
    if selected.weights_fingerprint != indexed_fingerprint(selected.weights):
        raise ValueError("Weights were modified after selection; select again explicitly.")
    if selected.integrity_fingerprint != selection_fingerprint(selected):
        raise ValueError("Selection configuration or reports changed after selection; select again.")
    if not selected.weights.index.equals(oof.columns):
        raise ValueError("Selected weight order must match the model columns.")
    weights = selected.weights.reindex(oof.columns)
    if weights.isna().any() or not test.columns.equals(oof.columns):
        raise ValueError("Model columns do not match the selected weights.")
    blended_oof = oof @ weights
    blended_test = test @ weights
    check_prediction(blended_oof, panel.train.index)
    check_prediction(blended_test, panel.test.index)
    if not np.isclose(weighted_directional_accuracy(panel.y, blended_oof), selected.selection_score, rtol=0, atol=1e-12):
        raise ValueError("Selected score does not match the exported predictions.")
    signs = pd.Series(np.where(blended_test >= 0, 1, -1), index=blended_test.index, name="prediction")
    summary = {
        "data_kind": "independent_synthetic_demo",
        "weights": weights.to_dict(),
        "selection_config": selected.config,
        "epsilon": float(selected.epsilon),
        "minimum_feasible_epsilon_on_grid": float(selected.min_feasible_epsilon),
        "oof_selection_score": weighted_directional_accuracy(panel.y, blended_oof),
        "warning": "Synthetic selection diagnostic, not a competition result or untouched holdout score.",
    }
    payloads = {
        "synthetic_oof_raw.csv": blended_oof.rename("prediction").to_csv(index_label="row_id").encode("utf-8"),
        "synthetic_test_raw.csv": blended_test.rename("prediction").to_csv(index_label="row_id").encode("utf-8"),
        "synthetic_submission.csv": signs.to_csv(index_label="row_id").encode("utf-8"),
        "synthetic_weight_candidates.csv": selected.candidates.to_csv(index=False).encode("utf-8"),
        "synthetic_weight_diagnostics.csv": selected.diagnostics.to_csv(index=False).encode("utf-8"),
        "synthetic_summary.json": json_bytes(summary),
    }
    manifest = {
        "schema_version": 2,
        "data_kind": "independent_synthetic_demo",
        "model": "ensemble",
        "panel_fingerprint": oof.attrs["shared_protocol"]["panel_fingerprint"],
        "shared_protocol": oof.attrs["shared_protocol"],
        "selection_config": selected.config,
        "selection_input_fingerprint": selected.input_fingerprint,
        "selection_result_fingerprint": selected.integrity_fingerprint,
        "selector_implementation_sha256": selected.selector_implementation_sha256,
        "selection_summary": summary,
        "weights": weights.to_dict(),
        "sources": {model: {
            "manifest_sha256": digest, "manifest": oof.attrs["source_records"][model],
        } for model, digest in oof.attrs["source_manifests"].items()},
    }
    committed = commit_bundle(Path(output_dir) / "ensemble", manifest, payloads)
    print(json.dumps(summary, indent=2))
    print("Published output generation:", committed["run_id"])
    print("Submission path:", (Path(output_dir) / "ensemble" / committed["files"]["synthetic_submission.csv"]["filename"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "ensemble", "all"), default="all",
                        help="ensemble requires existing artifacts from all three model runs.")
    parser.add_argument("--search", action="store_true", help="Run small demo hyperparameter searches.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--step", type=float, default=0.05, help="Simplex grid step for blend selection.")
    args = parser.parse_args()
    panel = make_synthetic_panel()
    for model_name in MODEL_NAMES:
        if args.model in (model_name, "all"):
            run_model(model_name, panel, args.output_dir, search=args.search)
    if args.model in ("ensemble", "all"):
        run_ensemble(panel, args.output_dir, epsilon=args.epsilon, step=args.step)


if __name__ == "__main__":
    main()
