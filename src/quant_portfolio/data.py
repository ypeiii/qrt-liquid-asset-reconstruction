"""Small artificial panels: no competition observations or private factors."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

METADATA_COLUMNS = ("day_id", "target_id", "pool_id")


@dataclass(frozen=True)
class Panel:
    """Aligned training rows, training outcomes, and unlabeled evaluation rows."""

    train: pd.DataFrame
    y: pd.Series
    test: pd.DataFrame

    def validate(self) -> None:
        if not self.train.index.equals(self.y.index):
            raise ValueError("Training rows and outcomes must have identical ordered IDs.")
        if self.train.empty or self.test.empty:
            raise ValueError("Both training and evaluation panels must be nonempty.")
        for name, frame in (("train", self.train), ("test", self.test)):
            if not frame.index.is_unique:
                raise ValueError(f"{name} row IDs must be unique.")
            if not set(METADATA_COLUMNS).issubset(frame.columns):
                raise ValueError(f"{name} is missing panel metadata.")
            if frame[list(METADATA_COLUMNS)].isna().any().any():
                raise ValueError(f"{name} metadata cannot contain missing values.")
        if not self.train.index.intersection(self.test.index).empty:
            raise ValueError("Training and evaluation row IDs must be disjoint.")
        if set(self.train.day_id).intersection(self.test.day_id):
            raise ValueError("Training and evaluation days must be disjoint.")
        if not set(self.test.target_id).issubset(self.train.target_id):
            raise ValueError("Evaluation targets must be present in training.")
        if not np.isfinite(self.y.to_numpy(dtype=float)).all():
            raise ValueError("Training outcomes must be finite.")
        target_pools = pd.concat([self.train, self.test]).groupby("target_id")["pool_id"]
        if (target_pools.nunique() != 1).any():
            raise ValueError("Every target must belong to exactly one pool.")


def panel_fingerprint(panel: Panel) -> str:
    """Hash ordered data, labels, IDs, and schema for local experiment identity.

    This is a change detector, not an anonymization mechanism. Keep fingerprints
    of private inputs in private artifacts. The pandas version is included
    because its internal hashing format is not promised stable across versions.
    """
    panel.validate()
    digest = hashlib.sha256()
    digest.update(f"panel-v1:pandas-{pd.__version__}".encode("utf-8"))
    for name, frame in (("train", panel.train), ("y", panel.y.to_frame()), ("test", panel.test)):
        schema = {
            "name": name,
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "index_dtype": str(frame.index.dtype),
            "index_names": [str(value) for value in frame.index.names],
            "shape": list(frame.shape),
        }
        digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def make_synthetic_panel(
    seed: int = 42,
    n_days: int = 60,
    n_test_days: int = 12,
    n_targets: int = 4,
    n_features: int = 8,
) -> Panel:
    """Generate a fresh teaching problem, unrelated to competition feature logic.

    Shared day observations, target-specific signals, and independent noise make
    the demo useful for grouped validation. Some covariates are missing on
    purpose. Test outcomes are neither retained nor exposed to model code.
    """
    dimensions = (n_days, n_test_days, n_targets, n_features)
    if any(not isinstance(value, int) or value < 1 for value in dimensions):
        raise ValueError("Panel dimensions must be positive integers.")
    if n_days < 5:
        raise ValueError("At least five training days are required.")
    generator = np.random.default_rng(seed)
    total_days = n_days + n_test_days
    day_id = np.repeat(np.arange(total_days), n_targets)
    target_id = np.tile(np.arange(n_targets), total_days)
    common = generator.normal(size=(total_days, n_features))
    values = common[day_id] + 0.35 * generator.normal(size=(len(day_id), n_features))
    coefficients = generator.normal(size=(n_targets, n_features)) / np.sqrt(n_features)
    signal = np.einsum("ij,ij->i", values, coefficients[target_id])
    response = signal + 0.45 * np.sin(values[:, 0]) + generator.normal(0, 0.8, len(day_id))
    values[generator.random(values.shape) < 0.04] = np.nan
    frame = pd.DataFrame(values, columns=[f"feature_{i:03d}" for i in range(n_features)])
    frame.insert(0, "pool_id", target_id % min(2, n_targets))
    frame.insert(0, "target_id", target_id)
    frame.insert(0, "day_id", day_id)
    frame.index = pd.Index([f"row_{i:06d}" for i in range(len(frame))], name="row_id")
    train_mask = frame.day_id < n_days
    train = frame.loc[train_mask].copy()
    test = frame.loc[~train_mask].copy()
    y = pd.Series(response[train_mask], index=train.index, name="outcome")
    result = Panel(train=train, y=y, test=test)
    result.validate()
    return result
