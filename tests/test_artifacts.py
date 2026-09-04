"""Artifact failures must be detected before prediction matrices are combined."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.artifacts import load_prediction_stack, save_model_artifacts
from quant_portfolio.data import make_synthetic_panel
from quant_portfolio.models import MODEL_NAMES


def prepare_artifacts(destination):
    panel = make_synthetic_panel(n_days=10, n_test_days=2, n_targets=2, n_features=3)
    for model in MODEL_NAMES:
        save_model_artifacts(destination, model, panel, panel.y.copy(), pd.Series(0.0, index=panel.test.index))
    return panel


def test_artifacts_roundtrip(tmp_path):
    panel = prepare_artifacts(tmp_path)
    oof, test = load_prediction_stack(tmp_path, panel)
    assert tuple(oof.columns) == MODEL_NAMES
    assert test.index.equals(panel.test.index)
    np.testing.assert_allclose(oof.ridge, panel.y)


def test_modified_prediction_rejected(tmp_path):
    panel = prepare_artifacts(tmp_path)
    path = tmp_path / "ridge" / "oof.csv"
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
