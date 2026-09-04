"""Fast tests exercise the public pipeline without any private data or factors."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.data import Panel, make_synthetic_panel, panel_fingerprint
from quant_portfolio.features import PrivateFeatureProvider, SyntheticFeatureProvider
from quant_portfolio.models import model_oof, refit_predict


@pytest.fixture
def small_panel():
    return make_synthetic_panel(n_days=15, n_test_days=3, n_targets=2, n_features=3)


def test_synthetic_generation_is_deterministic_and_disjoint(small_panel):
    again = make_synthetic_panel(n_days=15, n_test_days=3, n_targets=2, n_features=3)
    pd.testing.assert_frame_equal(small_panel.train, again.train)
    pd.testing.assert_series_equal(small_panel.y, again.y)
    assert not set(small_panel.train.day_id).intersection(small_panel.test.day_id)
    assert small_panel.train.filter(like="feature_").isna().any().any()


@pytest.mark.parametrize("model_name", ["ridge", "lightgbm", "extra_trees"])
def test_models_produce_aligned_complete_deterministic_predictions(small_panel, model_name):
    params = {} if model_name == "ridge" else {"n_estimators": 4}
    first = model_oof(model_name, small_panel, params)
    second = model_oof(model_name, small_panel, params)
    pd.testing.assert_series_equal(first, second)
    assert first.index.equals(small_panel.train.index)
    assert np.isfinite(first).all()
    test = refit_predict(model_name, small_panel, params)
    assert test.index.equals(small_panel.test.index)
    assert np.isfinite(test).all()


def test_provider_receives_only_training_labels_and_disjoint_days(small_panel):
    class AuditedProvider(SyntheticFeatureProvider):
        calls = 0

        def prepare(self, train_rows, train_y, evaluation_rows):
            self.calls += 1
            assert train_rows.index.equals(train_y.index)
            assert train_y.index.intersection(evaluation_rows.index).empty
            assert not set(train_rows.day_id).intersection(evaluation_rows.day_id)
            return super().prepare(train_rows, train_y, evaluation_rows)

    provider = AuditedProvider()
    model_oof("ridge", small_panel, provider=provider)
    assert provider.calls == 5
    refit_predict("ridge", small_panel, provider=provider)
    assert provider.calls == 6


def test_private_features_are_not_distributed(small_panel):
    with pytest.raises(NotImplementedError, match="not distributed"):
        model_oof("ridge", small_panel, provider=PrivateFeatureProvider())


def test_misaligned_provider_is_rejected(small_panel):
    class BrokenProvider(SyntheticFeatureProvider):
        def prepare(self, train_rows, train_y, evaluation_rows):
            fit, evaluation = super().prepare(train_rows, train_y, evaluation_rows)
            return fit.iloc[::-1], evaluation

    with pytest.raises(ValueError, match="ordered input IDs"):
        model_oof("ridge", small_panel, provider=BrokenProvider())


def test_panel_rejects_misaligned_labels(small_panel):
    broken = Panel(small_panel.train, small_panel.y.iloc[::-1], small_panel.test)
    with pytest.raises(ValueError, match="ordered IDs"):
        broken.validate()


def test_fingerprint_detects_values_order_schema_and_test_changes(small_panel):
    original = panel_fingerprint(small_panel)
    changed = Panel(small_panel.train.copy(), small_panel.y.copy(), small_panel.test.copy())
    changed.y.iloc[0] += 1
    assert panel_fingerprint(changed) != original
    changed = Panel(small_panel.train.copy(), small_panel.y.copy(), small_panel.test.copy())
    changed.test.iloc[0, -1] = 100
    assert panel_fingerprint(changed) != original
    changed = Panel(small_panel.train.iloc[::-1], small_panel.y.iloc[::-1], small_panel.test)
    assert panel_fingerprint(changed) != original
    changed = Panel(small_panel.train.copy(), small_panel.y.copy(), small_panel.test.copy())
    changed.train["feature_000"] = changed.train["feature_000"].astype("float32")
    assert panel_fingerprint(changed) != original


def test_ridge_handles_an_entirely_missing_feature(small_panel):
    small_panel.train["feature_000"] = np.nan
    small_panel.test["feature_000"] = np.nan
    assert np.isfinite(model_oof("ridge", small_panel)).all()


def test_unknown_model_is_rejected(small_panel):
    with pytest.raises(ValueError, match="Unknown model"):
        model_oof("unknown", small_panel)


def test_ridge_search_returns_a_reproducible_selection(small_panel):
    from quant_portfolio.search import ridge_grid_search

    result = ridge_grid_search(small_panel)
    assert set(result.params) == {"pool_alpha", "residual_alpha"}
    assert 0 <= result.score <= 1
    assert {"coarse", "fine"} == set(result.table.stage)
    assert len(result.table) == 12


def test_optuna_resume_preserves_an_unfinished_batch(small_panel, tmp_path, monkeypatch):
    import optuna
    import quant_portfolio.search as search

    # Test persistence quickly; model training is independently covered above.
    monkeypatch.setattr(search, "model_oof", lambda *args: small_panel.y.copy())
    path = tmp_path / "demo.sqlite3"
    first = search.run_optuna_search("extra_trees", small_panel, path, initial_trials=2)
    assert len(first.table) == 2
    study = optuna.load_study(study_name="synthetic_extra_trees", storage="sqlite:///" + path.as_posix())
    study.set_user_attr("active_finished_target", 3)
    second = search.run_optuna_search("extra_trees", small_panel, path)
    assert len(second.table) == 3
    third = search.run_optuna_search("extra_trees", small_panel, path, trials_per_reopen=2)
    assert len(third.table) == 5
    study._storage.remove_session()


def test_optuna_does_not_modify_existing_running_trials(small_panel, tmp_path):
    import optuna
    from quant_portfolio.search import run_optuna_search

    path = tmp_path / "running.sqlite3"
    study = optuna.create_study(
        study_name="synthetic_extra_trees", storage="sqlite:///" + path.as_posix(),
        direction="maximize",
    )
    active = study.ask()
    with pytest.raises(RuntimeError, match="No running trial was modified"):
        run_optuna_search("extra_trees", small_panel, path)
    assert study.trials[active.number].state == optuna.trial.TrialState.RUNNING
    study.tell(active, state=optuna.trial.TrialState.FAIL)
    study._storage.remove_session()


@pytest.mark.parametrize("change", ["labels", "features", "test", "seed", "folds"])
def test_optuna_rejects_changed_context_without_adding_trials(small_panel, tmp_path, monkeypatch, change):
    import optuna
    import quant_portfolio.search as search

    monkeypatch.setattr(search, "model_oof", lambda *args: small_panel.y.copy())
    path = tmp_path / "context.sqlite3"
    search.run_optuna_search("extra_trees", small_panel, path, initial_trials=1)
    kwargs = {}
    if change == "labels":
        small_panel.y.iloc[0] += 0.5
    elif change == "features":
        small_panel.train.iloc[0, -1] = 50
    elif change == "test":
        small_panel.test.iloc[0, -1] = 50
    elif change == "seed":
        kwargs["seed"] = 17
    elif change == "folds":
        kwargs["n_splits"] = 3
    with pytest.raises(ValueError, match="Search context changed"):
        search.run_optuna_search("extra_trees", small_panel, path, **kwargs)
    study = optuna.load_study(study_name="synthetic_extra_trees", storage="sqlite:///" + path.as_posix())
    assert len(study.trials) == 1
    assert study.user_attrs["active_finished_target"] == 1
    study._storage.remove_session()


def test_optuna_rejects_unversioned_history(small_panel, tmp_path):
    import optuna
    from quant_portfolio.search import run_optuna_search

    path = tmp_path / "unversioned.sqlite3"
    study = optuna.create_study(
        study_name="synthetic_extra_trees", storage="sqlite:///" + path.as_posix(),
        direction="maximize",
    )
    study.tell(study.ask(), 0.5)
    with pytest.raises(ValueError, match="unversioned"):
        run_optuna_search("extra_trees", small_panel, path)
    assert len(study.trials) == 1
    study._storage.remove_session()


def test_custom_search_provider_requires_explicit_version(small_panel, tmp_path):
    from quant_portfolio.search import run_optuna_search

    class UnversionedProvider(SyntheticFeatureProvider):
        pass

    with pytest.raises(ValueError, match="cache_key"):
        run_optuna_search("extra_trees", small_panel, tmp_path / "not_created.sqlite3", UnversionedProvider())
    assert not (tmp_path / "not_created.sqlite3").exists()


def test_custom_provider_version_change_is_rejected(small_panel, tmp_path, monkeypatch):
    import quant_portfolio.search as search

    class VersionedProvider(SyntheticFeatureProvider):
        cache_key = "example-v1"

    monkeypatch.setattr(search, "model_oof", lambda *args: small_panel.y.copy())
    path = tmp_path / "provider.sqlite3"
    provider = VersionedProvider()
    search.run_optuna_search("extra_trees", small_panel, path, provider, initial_trials=1)
    provider.cache_key = "example-v2"
    with pytest.raises(ValueError, match="Search context changed"):
        search.run_optuna_search("extra_trees", small_panel, path, provider)


@pytest.mark.parametrize("model_name", ["lightgbm", "extra_trees"])
def test_optuna_can_evaluate_a_real_tiny_model(small_panel, tmp_path, model_name):
    from quant_portfolio.blend import weighted_directional_accuracy
    from quant_portfolio.search import run_optuna_search

    result = run_optuna_search(model_name, small_panel, tmp_path / f"{model_name}.sqlite3", initial_trials=1)
    rebuilt = model_oof(model_name, small_panel, result.params)
    assert result.score == pytest.approx(weighted_directional_accuracy(small_panel.y, rebuilt))
    assert result.table.state.tolist() == ["COMPLETE"]
