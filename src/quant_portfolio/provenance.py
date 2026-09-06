"""Local experiment identity for generated predictions, never a secrecy tool."""

import hashlib
from importlib.metadata import version
import json
from numbers import Integral
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import Panel, panel_fingerprint
from .features import SyntheticFeatureProvider


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def indexed_fingerprint(*objects) -> str:
    """Hash ordered values and schema; do not publish private input digests."""
    digest = hashlib.sha256()
    for value in objects:
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        schema = {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "index_dtype": str(frame.index.dtype),
            "index_names": [str(name) for name in frame.index.names],
            "shape": list(frame.shape),
        }
        digest.update(json_bytes(schema))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def generation_context(panel: Panel, model_name: str, params: dict, provider, seed: int) -> dict:
    """Capture actual model-call settings, runtime and public source versions."""
    if not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    source = Path(__file__).parent
    code_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("data.py", "features.py", "models.py", "provenance.py")
    }
    return {
        "schema_version": 1,
        "panel_fingerprint": panel_fingerprint(panel),
        "model": model_name,
        "params": json.loads(json_bytes(params)),
        "seed": int(seed),
        "provider": {
            "type": f"{type(provider).__module__}.{type(provider).__qualname__}",
            "cache_key": "synthetic-column-selector-v1" if type(provider) is SyntheticFeatureProvider
            else getattr(provider, "cache_key", None),
        },
        "runtime_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
        },
        "implementation_hashes": code_hashes,
    }


def fold_protocol(panel: Panel, n_splits: int) -> dict:
    if not isinstance(n_splits, Integral) or isinstance(n_splits, bool):
        raise ValueError("n_splits must be an integer.")
    if not 2 <= n_splits <= panel.train.day_id.nunique():
        raise ValueError("n_splits must be between two and the number of training days.")
    assignments = pd.Series(-1, index=panel.train.index, name="oof_fold", dtype="int64")
    for fold, (_, positions) in enumerate(GroupKFold(n_splits=int(n_splits)).split(
        panel.train, groups=panel.train.day_id,
    )):
        assignments.iloc[positions] = fold
    return {
        "splitter": "GroupKFold",
        "shuffle": False,
        "n_splits": int(n_splits),
        "assignments_sha256": indexed_fingerprint(assignments),
    }


def attach_prediction_context(prediction: pd.Series, context: dict, protocol: dict | None) -> None:
    prediction.attrs["provenance"] = {
        "context": context,
        "kind": "oof" if protocol is not None else "full_refit",
        "fold_protocol": protocol,
        "values_sha256": indexed_fingerprint(prediction.rename("prediction")),
    }


def validate_prediction_context(prediction: pd.Series, kind: str) -> dict:
    record = prediction.attrs.get("provenance")
    if not isinstance(record, dict) or record.get("kind") != kind:
        raise ValueError("Prediction provenance is missing; regenerate with model_oof/refit_predict.")
    if record.get("values_sha256") != indexed_fingerprint(prediction.rename("prediction")):
        raise ValueError("Predictions changed after generation; regenerate their provenance.")
    context = record.get("context", {})
    key = context.get("provider", {}).get("cache_key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Persisted predictions need a nonempty provider cache_key.")
    return record


def shared_protocol(context: dict, protocol: dict) -> dict:
    """The public workflow deliberately requires a shared experiment protocol."""
    return {
        key: context[key] for key in (
            "panel_fingerprint", "seed", "provider", "runtime_versions", "implementation_hashes",
        )
    } | {"fold_protocol": protocol}
