"""Public feature contract; the private implementation is intentionally absent."""

from typing import Protocol

import numpy as np
import pandas as pd


class FeatureProvider(Protocol):
    """Prepare a single fit/evaluation pair without receiving evaluation labels.

    Implementations must preserve row IDs and column order. Any learned
    transform must use this call's training rows only. If training labels are
    used to construct features, the returned training features must themselves
    be cross-fitted within those training groups. This interface documents that
    obligation; it cannot prove a third-party implementation follows it.
    Persisted searches additionally require a custom provider's nonempty
    ``cache_key`` string to change whenever its implementation or settings change.
    """

    def prepare(
        self,
        train_rows: pd.DataFrame,
        train_y: pd.Series,
        evaluation_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class SyntheticFeatureProvider:
    """Select pre-generated artificial covariates; learn no transform or factors."""

    def prepare(
        self,
        train_rows: pd.DataFrame,
        train_y: pd.Series,
        evaluation_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not train_rows.index.equals(train_y.index):
            raise ValueError("Training labels must be aligned before feature preparation.")
        columns = [column for column in train_rows if column.startswith("feature_")]
        if not columns or not set(columns).issubset(evaluation_rows.columns):
            raise ValueError("Synthetic covariates are missing or inconsistent.")
        return train_rows[columns].copy(), evaluation_rows[columns].copy()


class PrivateFeatureProvider:
    """An integration placeholder, not an implementation or an encrypted factor."""

    def prepare(
        self,
        train_rows: pd.DataFrame,
        train_y: pd.Series,
        evaluation_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError(
            "Proprietary feature construction is not distributed. "
            "Use SyntheticFeatureProvider for the public executable demo."
        )


def validate_feature_pair(
    train_features: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    train_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
) -> None:
    """Fail early on alignment errors instead of silently producing a wrong OOF."""
    for features, rows in (
        (train_features, train_rows),
        (evaluation_features, evaluation_rows),
    ):
        if not isinstance(features, pd.DataFrame) or not features.index.equals(rows.index):
            raise ValueError("Feature rows must retain the exact ordered input IDs.")
        if not features.columns.is_unique or features.shape[1] == 0:
            raise ValueError("Feature names must be unique and nonempty.")
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise ValueError("Features can contain NaN, but not infinity.")
    if not train_features.columns.equals(evaluation_features.columns):
        raise ValueError("Fit/evaluation feature names and their order must match.")
