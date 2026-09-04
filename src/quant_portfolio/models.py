"""Grouped OOF evaluation and full refits of three deliberately small models."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Panel
from .features import FeatureProvider, SyntheticFeatureProvider, validate_feature_pair

MODEL_NAMES = ("ridge", "lightgbm", "extra_trees")


def _ridge(alpha: float):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge regularization must be finite and positive.")
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def _tree(model_name: str, params: Mapping[str, Any], seed: int):
    forbidden = {"random_state", "n_jobs"}.intersection(params)
    if forbidden:
        raise ValueError(f"Use the public seed argument; fixed execution settings: {forbidden}")
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        settings = dict(
            n_estimators=48, learning_rate=0.08, num_leaves=9,
            min_child_samples=5, verbosity=-1, force_col_wise=True,
        )
        settings.update(params)
        return LGBMRegressor(random_state=seed, n_jobs=1, **settings)
    settings = dict(n_estimators=48, max_depth=5, min_samples_leaf=2, max_features=0.9)
    settings.update(params)
    model = ExtraTreesRegressor(random_state=seed, n_jobs=1, **settings)
    if not model.__sklearn_tags__().input_tags.allow_nan:
        raise RuntimeError("This demo requires an ExtraTrees version with native NaN support.")
    return model


def _fit_predict(
    model_name: str,
    train_rows: pd.DataFrame,
    train_y: pd.Series,
    evaluation_rows: pd.DataFrame,
    params: Mapping[str, Any],
    provider: FeatureProvider,
    seed: int,
) -> pd.Series:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {MODEL_NAMES}.")
    fit_features, evaluation_features = provider.prepare(train_rows, train_y, evaluation_rows)
    validate_feature_pair(fit_features, evaluation_features, train_rows, evaluation_rows)
    prediction = pd.Series(np.nan, index=evaluation_rows.index, name="prediction")
    base_fit = pd.Series(0.0, index=train_rows.index)
    base_evaluation = pd.Series(0.0, index=evaluation_rows.index)

    if model_name == "ridge":
        unexpected = set(params).difference({"pool_alpha", "residual_alpha"})
        if unexpected:
            raise ValueError(f"Unsupported Ridge parameters: {unexpected}")
        for pool, rows in train_rows.groupby("pool_id", sort=True):
            fitted = _ridge(float(params.get("pool_alpha", 1.0)))
            fitted.fit(fit_features.loc[rows.index], train_y.loc[rows.index])
            base_fit.loc[rows.index] = fitted.predict(fit_features.loc[rows.index])
            eval_ids = evaluation_rows.index[evaluation_rows.pool_id.eq(pool)]
            if len(eval_ids):
                base_evaluation.loc[eval_ids] = fitted.predict(evaluation_features.loc[eval_ids])
        if not set(evaluation_rows.pool_id).issubset(train_rows.pool_id):
            raise ValueError("Every evaluation pool must appear in the fit partition.")

    for target, rows in evaluation_rows.groupby("target_id", sort=True):
        fit_ids = train_rows.index[train_rows.target_id.eq(target)]
        if not len(fit_ids):
            raise ValueError(f"Target {target!r} has no observations in the fit partition.")
        if model_name == "ridge":
            fitted = _ridge(float(params.get("residual_alpha", 1.0)))
            fitted.fit(fit_features.loc[fit_ids], train_y.loc[fit_ids] - base_fit.loc[fit_ids])
            values = fitted.predict(evaluation_features.loc[rows.index])
            prediction.loc[rows.index] = values + base_evaluation.loc[rows.index]
        else:
            fitted = _tree(model_name, params, seed)
            fitted.fit(fit_features.loc[fit_ids], train_y.loc[fit_ids])
            prediction.loc[rows.index] = fitted.predict(evaluation_features.loc[rows.index])
    if not np.isfinite(prediction.to_numpy()).all():
        raise RuntimeError("Predictions must cover every evaluation row with finite values.")
    return prediction


def model_oof(
    model_name: str,
    panel: Panel,
    params: Mapping[str, Any] | None = None,
    provider: FeatureProvider | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Return one out-of-fold prediction per row, holding out entire days.

    This is grouped validation, not chronological walk-forward validation.
    Hyperparameters selected using this OOF need a separate holdout for an
    unbiased estimate of the complete selection procedure.
    """
    panel.validate()
    feature_provider = provider if provider is not None else SyntheticFeatureProvider()
    output = pd.Series(np.nan, index=panel.train.index, name="prediction")
    coverage = pd.Series(0, index=panel.train.index)
    splitter = GroupKFold(n_splits=n_splits)
    for train_positions, evaluation_positions in splitter.split(panel.train, groups=panel.train.day_id):
        fit_rows = panel.train.iloc[train_positions]
        evaluation_rows = panel.train.iloc[evaluation_positions]
        if set(fit_rows.day_id).intersection(evaluation_rows.day_id):
            raise RuntimeError("A validation day was also present in training.")
        values = _fit_predict(
            model_name, fit_rows, panel.y.loc[fit_rows.index], evaluation_rows,
            dict(params or {}), feature_provider, seed,
        )
        output.loc[values.index] = values
        coverage.loc[values.index] += 1
    if not coverage.eq(1).all() or not np.isfinite(output.to_numpy()).all():
        raise RuntimeError("OOF predictions must cover every training row exactly once.")
    return output


def refit_predict(
    model_name: str,
    panel: Panel,
    params: Mapping[str, Any] | None = None,
    provider: FeatureProvider | None = None,
    seed: int = 42,
) -> pd.Series:
    """Refit on all training rows and predict the disjoint unlabeled test panel."""
    panel.validate()
    return _fit_predict(
        model_name, panel.train, panel.y, panel.test, dict(params or {}),
        provider if provider is not None else SyntheticFeatureProvider(), seed,
    )
