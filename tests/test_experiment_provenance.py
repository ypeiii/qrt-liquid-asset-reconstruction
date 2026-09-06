"""Regression checks for source identity, exact export and interrupted reruns."""

import json

import numpy as np
import pandas as pd
import pytest

import quant_portfolio.artifacts as artifacts
import quant_portfolio.demo as demo
from quant_portfolio.blend import learn_common_weights
from quant_portfolio.data import make_synthetic_panel
from quant_portfolio.models import MODEL_NAMES, model_oof, refit_predict


@pytest.fixture
def panel():
    return make_synthetic_panel(n_days=10, n_test_days=2, n_targets=2, n_features=3)


def save_model(root, panel, name, splits=5, seed=42, params=None):
    params = ({} if name == "ridge" else {"n_estimators": 2}) if params is None else params
    return artifacts.save_model_artifacts(
        root, name, panel, model_oof(name, panel, params, n_splits=splits, seed=seed),
        refit_predict(name, panel, params, seed=seed), params,
    )


def save_stack(root, panel):
    for name in MODEL_NAMES:
        save_model(root, panel, name)


def selection(root, panel, step=0.1):
    oof, _ = artifacts.load_prediction_stack(root, panel)
    return learn_common_weights(panel.y, oof, panel.train.day_id, step=step, epsilon=1)


def manifest(root, model):
    return json.loads((root / model / "manifest.json").read_text(encoding="utf-8"))


def output_path(root, model, filename):
    return root / model / manifest(root, model)["files"][filename]["filename"]


@pytest.mark.parametrize("change", ["splits", "seed"])
def test_mixed_protocol_rejected(tmp_path, panel, change):
    save_stack(tmp_path, panel)
    save_model(tmp_path, panel, "ridge", splits=3 if change == "splits" else 5,
               seed=43 if change == "seed" else 42)
    with pytest.raises(ValueError, match="experiment protocols differ"):
        artifacts.load_prediction_stack(tmp_path, panel)


def test_generation_context_records_actual_settings(tmp_path, panel):
    result = save_model(tmp_path, panel, "ridge", splits=3, seed=7, params={"pool_alpha": 2.0})
    context = result["generation_context"]
    assert context["seed"] == 7
    assert context["params"] == {"pool_alpha": 2.0}
    assert result["fold_protocol"]["n_splits"] == 3
    assert context["provider"]["cache_key"] == "synthetic-column-selector-v1"
    assert "python" in context["runtime_versions"]
    assert "models.py" in context["implementation_hashes"]


def test_modified_predictions_cannot_keep_original_provenance(tmp_path, panel):
    oof = model_oof("ridge", panel)
    oof.iloc[0] += 0.1
    with pytest.raises(ValueError, match="changed after generation"):
        artifacts.save_model_artifacts(tmp_path, "ridge", panel, oof, refit_predict("ridge", panel))
    assert not (tmp_path / "ridge").exists()


def test_unannotated_predictions_are_rejected(tmp_path, panel):
    with pytest.raises(ValueError, match="provenance is missing"):
        artifacts.save_model_artifacts(tmp_path, "ridge", panel, panel.y.copy(),
                                       pd.Series(0.0, index=panel.test.index))


def test_oof_and_refit_must_share_generation_context(tmp_path, panel):
    with pytest.raises(ValueError, match="generation contexts do not match"):
        artifacts.save_model_artifacts(tmp_path, "ridge", panel,
                                       model_oof("ridge", panel, seed=42),
                                       refit_predict("ridge", panel, seed=43))


def test_saved_params_cannot_differ_from_model_call(tmp_path, panel):
    with pytest.raises(ValueError, match="Saved parameters"):
        artifacts.save_model_artifacts(tmp_path, "ridge", panel,
                                       model_oof("ridge", panel), refit_predict("ridge", panel),
                                       {"pool_alpha": 5.0})


def test_old_schema_has_actionable_migration_error(tmp_path, panel):
    destination = tmp_path / "ridge"
    destination.mkdir()
    (destination / "manifest.json").write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="regenerate all three models in a fresh"):
        artifacts.load_prediction_stack(tmp_path, panel)


def test_missing_inputs_have_actionable_message(tmp_path, panel):
    with pytest.raises(ValueError, match="--model all first"):
        demo.run_ensemble(panel, tmp_path)
    assert not (tmp_path / "ensemble").exists()


def test_existing_selection_export_does_not_search_again(tmp_path, panel, monkeypatch):
    save_stack(tmp_path, panel)
    chosen = selection(tmp_path, panel, step=0.1)
    def unexpected_search(*args, **kwargs):
        raise AssertionError("Export must not run another weight search.")
    monkeypatch.setattr(demo, "learn_common_weights", unexpected_search)
    exported = demo.run_ensemble(panel, tmp_path, selected=chosen)
    assert exported is chosen
    recorded = artifacts.verify_ensemble_artifacts(tmp_path, panel)
    assert recorded["selection_config"]["step"] == 0.1
    assert recorded["weights"] == chosen.weights.to_dict()
    assert len(pd.read_csv(output_path(tmp_path, "ensemble", "synthetic_weight_candidates.csv"))) == 66


def test_export_supplied_step_is_honored_without_existing_selection(tmp_path, panel):
    save_stack(tmp_path, panel)
    exported = demo.run_ensemble(panel, tmp_path, step=0.1, epsilon=1)
    assert exported.config["step"] == 0.1
    assert artifacts.verify_ensemble_artifacts(tmp_path, panel)["selection_config"]["step"] == 0.1


def test_conflicting_export_config_rejected_before_output(tmp_path, panel):
    save_stack(tmp_path, panel)
    with pytest.raises(ValueError, match="Export controls differ"):
        demo.run_ensemble(panel, tmp_path, selected=selection(tmp_path, panel), step=0.05)
    assert not (tmp_path / "ensemble").exists()


def test_changed_model_generation_requires_new_selection(tmp_path, panel):
    save_stack(tmp_path, panel)
    chosen = selection(tmp_path, panel)
    save_model(tmp_path, panel, "ridge")  # Same predictions, new audited input generation.
    with pytest.raises(ValueError, match="Model generations changed"):
        demo.run_ensemble(panel, tmp_path, selected=chosen)
    assert not (tmp_path / "ensemble").exists()


def test_mutated_weights_rejected_before_output(tmp_path, panel):
    save_stack(tmp_path, panel)
    chosen = selection(tmp_path, panel)
    chosen.weights.iloc[:] = [1.0, -1.0, 1.0]
    with pytest.raises(ValueError, match="Weights were modified"):
        demo.run_ensemble(panel, tmp_path, selected=chosen)
    assert not (tmp_path / "ensemble").exists()


@pytest.mark.parametrize("field", ["config", "candidates", "diagnostics"])
def test_mutated_selection_reports_rejected_before_output(tmp_path, panel, field):
    save_stack(tmp_path, panel)
    chosen = selection(tmp_path, panel)
    if field == "config":
        chosen.config["step"] = 0.2
    elif field == "candidates":
        chosen.candidates["full_oof"] = 999.0
    else:
        chosen.diagnostics["selected_regret"] = 999.0
    with pytest.raises(ValueError, match="reports changed after selection"):
        demo.run_ensemble(panel, tmp_path, selected=chosen)
    assert not (tmp_path / "ensemble").exists()


@pytest.mark.parametrize("model", ["ridge", "ensemble"])
def test_interrupted_publication_keeps_old_generation(tmp_path, panel, monkeypatch, model):
    save_stack(tmp_path, panel)
    demo.run_ensemble(panel, tmp_path, epsilon=1)
    before = (tmp_path / model / "manifest.json").read_bytes()
    def interrupted(*args):
        raise OSError("Simulated interruption before manifest publication")
    monkeypatch.setattr(artifacts, "_replace_manifest", interrupted)
    with pytest.raises(OSError, match="Simulated interruption"):
        if model == "ridge":
            save_model(tmp_path, panel, model, params={"pool_alpha": 2.0})
        else:
            demo.run_ensemble(panel, tmp_path, epsilon=1, step=0.1)
    assert (tmp_path / model / "manifest.json").read_bytes() == before
    artifacts.load_prediction_stack(tmp_path, panel)
    artifacts.verify_ensemble_artifacts(tmp_path, panel)


def test_saved_ensemble_remains_verifiable_after_base_rerun(tmp_path, panel):
    save_stack(tmp_path, panel)
    demo.run_ensemble(panel, tmp_path, epsilon=1)
    saved = artifacts.verify_ensemble_artifacts(tmp_path, panel)
    save_model(tmp_path, panel, "ridge", params={"pool_alpha": 5.0})
    assert artifacts.verify_ensemble_artifacts(tmp_path, panel) == saved


@pytest.mark.parametrize("kind", ["output", "source"])
def test_ensemble_verifier_rejects_tampered_files(tmp_path, panel, kind):
    save_stack(tmp_path, panel)
    demo.run_ensemble(panel, tmp_path, epsilon=1)
    path = output_path(tmp_path, "ensemble", "synthetic_test_raw.csv") if kind == "output" else output_path(tmp_path, "ridge", "oof.csv")
    path.write_bytes(path.read_bytes() + b"unexpected,1\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.verify_ensemble_artifacts(tmp_path, panel)
