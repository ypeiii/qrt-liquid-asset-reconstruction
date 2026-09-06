"""Artifact failures must be detected before prediction matrices are combined."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.artifacts import load_prediction_stack, save_model_artifacts
from quant_portfolio.data import make_synthetic_panel
from quant_portfolio.models import MODEL_NAMES, model_oof, refit_predict


def prepare_artifacts(destination):
    panel = make_synthetic_panel(n_days=10, n_test_days=2, n_targets=2, n_features=3)
    for model in MODEL_NAMES:
        params = {} if model == "ridge" else {"n_estimators": 2}
        save_model_artifacts(destination, model, panel,
                             model_oof(model, panel, params), refit_predict(model, panel, params), params)
    return panel


def test_artifacts_roundtrip(tmp_path):
    panel = prepare_artifacts(tmp_path)
    oof, test = load_prediction_stack(tmp_path, panel)
    assert tuple(oof.columns) == MODEL_NAMES
    assert test.index.equals(panel.test.index)
    np.testing.assert_allclose(oof.ridge, model_oof("ridge", panel), rtol=1e-12, atol=1e-14)
    assert set(oof.attrs["source_manifests"]) == set(MODEL_NAMES)


def test_modified_prediction_rejected(tmp_path):
    panel = prepare_artifacts(tmp_path)
    manifest = json.loads((tmp_path / "ridge" / "manifest.json").read_text())
    path = tmp_path / "ridge" / manifest["files"]["oof.csv"]["filename"]
    path.write_text(path.read_text() + "unexpected,1\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prediction_stack(tmp_path, panel)


def test_changed_data_rejected(tmp_path):
    panel = prepare_artifacts(tmp_path)
    panel.y.iloc[0] += 1.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_prediction_stack(tmp_path, panel)


def test_misordered_predictions_rejected(tmp_path):
    panel = make_synthetic_panel()
    with pytest.raises(ValueError, match="ordered IDs"):
        save_model_artifacts(tmp_path, "ridge", panel, panel.y.iloc[::-1], pd.Series(0.0, index=panel.test.index))
