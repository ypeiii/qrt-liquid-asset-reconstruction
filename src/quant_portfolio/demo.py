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

from .artifacts import check_prediction, load_prediction_stack, save_model_artifacts
from .blend import learn_common_weights, weighted_directional_accuracy
from .data import Panel, make_synthetic_panel
from .models import MODEL_NAMES, model_oof, refit_predict
from .search import ridge_grid_search, run_optuna_search


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


def run_ensemble(panel: Panel, output_dir: str | Path, epsilon: float = 0.05):
    """Use the same selected weights for aligned OOF and full-refit predictions."""
    oof, test = load_prediction_stack(output_dir, panel)
    selected = learn_common_weights(panel.y, oof, panel.train.day_id, epsilon=epsilon)
    weights = selected.weights.reindex(oof.columns)
    if weights.isna().any() or not test.columns.equals(oof.columns):
        raise ValueError("Model columns do not match the selected weights.")
    blended_oof = oof @ weights
    blended_test = test @ weights
    check_prediction(blended_oof, panel.train.index)
    check_prediction(blended_test, panel.test.index)
    destination = Path(output_dir) / "ensemble"
    destination.mkdir(parents=True, exist_ok=True)
    blended_oof.rename("prediction").to_csv(destination / "synthetic_oof_raw.csv", index_label="row_id")
    blended_test.rename("prediction").to_csv(destination / "synthetic_test_raw.csv", index_label="row_id")
    signs = pd.Series(np.where(blended_test >= 0, 1, -1), index=blended_test.index, name="prediction")
    signs.to_csv(destination / "synthetic_submission.csv", index_label="row_id")
    restored = pd.read_csv(destination / "synthetic_submission.csv", index_col="row_id").prediction
    pd.testing.assert_series_equal(restored, signs, check_names=False)
    selected.candidates.to_csv(destination / "synthetic_weight_candidates.csv", index=False)
    selected.diagnostics.to_csv(destination / "synthetic_weight_diagnostics.csv", index=False)
    summary = {
        "data_kind": "independent_synthetic_demo",
        "weights": weights.to_dict(),
        "epsilon": float(selected.epsilon),
        "minimum_feasible_epsilon_on_grid": float(selected.min_feasible_epsilon),
        "oof_selection_score": weighted_directional_accuracy(panel.y, blended_oof),
        "warning": "Synthetic selection diagnostic, not a competition result or untouched holdout score.",
    }
    (destination / "synthetic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "ensemble", "all"), default="all")
    parser.add_argument("--search", action="store_true", help="Run small demo hyperparameter searches.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()
    panel = make_synthetic_panel()
    for model_name in MODEL_NAMES:
        if args.model in (model_name, "all"):
            run_model(model_name, panel, args.output_dir, search=args.search)
    if args.model in ("ensemble", "all"):
        run_ensemble(panel, args.output_dir, epsilon=args.epsilon)


if __name__ == "__main__":
    main()

