"""Small demo searches with explicit selection and restart limitations."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from .blend import weighted_directional_accuracy
from .data import Panel, panel_fingerprint
from .features import FeatureProvider, SyntheticFeatureProvider
from .models import model_oof


@dataclass(frozen=True)
class SearchResult:
    params: dict[str, Any]
    score: float
    table: pd.DataFrame


def ridge_grid_search(
    panel: Panel,
    provider: FeatureProvider | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> SearchResult:
    """Try a tiny coarse grid, then an absolute 0.01 local step in both alphas."""
    rows = []
    visited = set()

    def evaluate(pool_alpha: float, residual_alpha: float, stage: str) -> None:
        key = (round(pool_alpha, 8), round(residual_alpha, 8))
        if key in visited:
            return
        visited.add(key)
        params = dict(pool_alpha=key[0], residual_alpha=key[1])
        oof = model_oof("ridge", panel, params, provider, n_splits, seed)
        rows.append({**params, "score": weighted_directional_accuracy(panel.y, oof), "stage": stage})

    for pool_alpha in (0.3, 1.5):
        for residual_alpha in (0.3, 1.5):
            evaluate(pool_alpha, residual_alpha, "coarse")
    coarse_best = max(rows, key=lambda record: record["score"])
    for pool_delta in (-0.01, 0.0, 0.01):
        for residual_delta in (-0.01, 0.0, 0.01):
            evaluate(
                coarse_best["pool_alpha"] + pool_delta,
                coarse_best["residual_alpha"] + residual_delta,
                "fine",
            )
    table = pd.DataFrame(rows).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    best = table.iloc[0]
    return SearchResult(
        params={key: float(best[key]) for key in ("pool_alpha", "residual_alpha")},
        score=float(best["score"]), table=table,
    )


def run_optuna_search(
    model_name: str,
    panel: Panel,
    storage_path: str | Path,
    provider: FeatureProvider | None = None,
    initial_trials: int = 3,
    trials_per_reopen: int = 2,
    n_splits: int = 5,
    seed: int = 42,
    study_name: str | None = None,
) -> SearchResult:
    """Resume an unfinished batch; add a new small batch after it finishes.

    SQLite stores trial history and the active batch target, not the sampler's
    RNG state. A fixed seed therefore does not promise the same proposal stream
    across process restarts. Do not open the same study in concurrent kernels.
    Existing RUNNING trials block execution; they are never auto-failed here.
    An interrupted trial cannot resume halfway through a fold.
    Data or search-context changes require a new study. Custom feature providers
    must expose a nonempty cache_key identifying their implementation/config.
    """
    import optuna
    from optuna.trial import TrialState

    if model_name not in ("lightgbm", "extra_trees"):
        raise ValueError("Optuna demo supports lightgbm and extra_trees only.")
    if initial_trials < 1 or trials_per_reopen < 1:
        raise ValueError("Trial budgets must be positive.")
    panel.validate()
    feature_provider = provider if provider is not None else SyntheticFeatureProvider()
    if type(feature_provider) is SyntheticFeatureProvider:
        provider_key = "synthetic-column-selector-v1"
    else:
        provider_key = getattr(feature_provider, "cache_key", None)
        if not isinstance(provider_key, str) or not provider_key.strip():
            raise ValueError("Custom providers need a nonempty cache_key for persisted searches.")
    context = {
        "schema_version": 1,
        "panel_fingerprint": panel_fingerprint(panel),
        "provider_type": f"{type(feature_provider).__module__}.{type(feature_provider).__qualname__}",
        "provider_cache_key": provider_key,
        "n_splits": n_splits,
        "seed": seed,
        "model_name": model_name,
        "search_space_version": "synthetic-tree-search-v1",
        "runtime_versions": {
            package: version(package) for package in ("numpy", "scikit-learn", "lightgbm", "optuna")
        },
    }
    path = Path(storage_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///" + path.as_posix(),
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    try:
        study = optuna.create_study(
            study_name=study_name or f"synthetic_{model_name}", storage=storage,
            direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=2),
        )
        running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if running:
            raise RuntimeError(
                "Study has RUNNING trials; confirm their owner has stopped before "
                "resolving them explicitly. No running trial was modified."
            )
        if study.direction != optuna.study.StudyDirection.MAXIMIZE:
            raise ValueError("The persisted study must maximize the selection score.")
        saved_context = study.user_attrs.get("search_context")
        if saved_context is None and study.get_trials(deepcopy=False):
            raise ValueError("Existing trial history is unversioned; use a new study name or database.")
        if saved_context is not None and saved_context != context:
            raise ValueError("Search context changed; use a new study name or database. No trials were added.")
        if saved_context is None:
            study.set_user_attr("search_context", context)
        counted = (TrialState.COMPLETE, TrialState.PRUNED)
        finished = len(study.get_trials(deepcopy=False, states=counted))
        saved_target = study.user_attrs.get("active_finished_target")
        target = (
            int(saved_target) if saved_target is not None and finished < int(saved_target)
            else (initial_trials if finished == 0 else finished + trials_per_reopen)
        )
        study.set_user_attr("active_finished_target", target)
        study.set_user_attr("sampler_seed", seed)
        study.set_user_attr("data_kind", "synthetic_demo")

        def objective(trial):
            if model_name == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 24, 64, step=8),
                    "num_leaves": trial.suggest_int("num_leaves", 5, 13, step=2),
                    "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12),
                    "min_child_samples": trial.suggest_int("min_child_samples", 3, 9),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 24, 64, step=8),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                    "max_features": trial.suggest_float("max_features", 0.65, 1.0),
                }
            predictions = model_oof(model_name, panel, params, feature_provider, n_splits, seed)
            return weighted_directional_accuracy(panel.y, predictions)

        # Unexpected failures propagate, leaving the same persisted target for
        # the next run. This is batch continuation, not mid-trial checkpointing.
        study.optimize(objective, n_trials=target - finished, n_jobs=1)
        best = study.best_trial
        table = study.trials_dataframe()
        return SearchResult(params=dict(best.params), score=float(best.value), table=table)
    finally:
        storage.remove_session()
        storage.engine.dispose()
