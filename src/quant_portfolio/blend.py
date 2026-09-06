"""Raw-scale, three-model blend selection for a synthetic research example.

All optima here are over a finite simplex grid, not a continuous or global
optimum. Repeated group splits reuse the same base-model OOF predictions: they
measure weight-selection stability, not independent end-to-end validation.
"""

from dataclasses import dataclass, replace
import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import indexed_fingerprint, json_bytes


_WEIGHT_COLUMNS = ["weight_0", "weight_1", "weight_2"]
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CommonWeightsResult:
    """Selected grid weights and explicitly selection-biased diagnostics."""

    weights: pd.Series
    candidates: pd.DataFrame
    diagnostics: pd.DataFrame
    epsilon: float
    min_feasible_epsilon: float
    selection_score: float
    config: dict
    input_fingerprint: str
    source_manifests: dict
    weights_fingerprint: str
    selector_implementation_sha256: str
    integrity_fingerprint: str = ""


def selection_fingerprint(result: CommonWeightsResult) -> str:
    """Detect accidental edits to a selected result before publication."""
    digest = hashlib.sha256(json_bytes({
        "config": result.config, "epsilon": result.epsilon,
        "min_feasible_epsilon": result.min_feasible_epsilon, "score": result.selection_score,
        "input_fingerprint": result.input_fingerprint, "source_manifests": result.source_manifests,
        "selector_implementation_sha256": result.selector_implementation_sha256,
    }))
    digest.update(indexed_fingerprint(result.weights, result.candidates, result.diagnostics).encode("ascii"))
    return digest.hexdigest()


def _numeric_vector(values, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _target_weights(y: np.ndarray) -> np.ndarray:
    # Rescale only metric weights, not predictions. This avoids overflow when
    # summing large finite targets and leaves the metric mathematically intact.
    absolute = np.abs(y)
    scale = absolute.max()
    if scale == 0:
        raise ValueError("The total absolute target weight must be positive.")
    scaled = absolute / scale
    return scaled / scaled.sum()


def weighted_directional_accuracy(y, p) -> float:
    """Return abs(y)-weighted sign agreement; zero belongs to sign >= 0.

    Inputs must be equally sized, finite, nonempty vectors. If both inputs are
    Series their indices must match exactly. All-zero targets are undefined.
    """
    target = _numeric_vector(y, "y")
    prediction = _numeric_vector(p, "p")
    if target.shape != prediction.shape:
        raise ValueError("y and p must have the same shape.")
    if isinstance(y, pd.Series) and isinstance(p, pd.Series) and not y.index.equals(p.index):
        raise ValueError("y and p indices must be aligned in the same order.")
    agreement = (target >= 0) == (prediction >= 0)
    return float(np.clip(np.sum(_target_weights(target) * agreement), 0, 1))


def simplex_grid(step: float = 0.05) -> np.ndarray:
    """Enumerate three nonnegative weights summing to one, in lexical order.

    The positive step must divide one (within floating-point tolerance).
    Integer coordinates avoid an accumulating floating-point stepping error.
    """
    if not np.isscalar(step):
        raise ValueError("step must be a finite positive scalar that divides one.")
    try:
        step = float(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("step must be a finite positive scalar that divides one.") from exc
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("step must be finite and in (0, 1].")
    inverse = 1.0 / step
    if not np.isfinite(inverse):
        raise ValueError("step is too small to form a finite grid.")
    divisions = round(inverse)
    if not np.isclose(inverse, divisions, rtol=0, atol=1e-10):
        raise ValueError("step must divide one exactly, such as 0.05, 0.1, or 0.25.")
    return np.asarray(
        [(i, j, divisions - i - j) for i in range(divisions + 1)
         for j in range(divisions - i + 1)],
        dtype=float,
    ) / divisions


def _validate_inputs(y, oof, groups):
    if not isinstance(y, pd.Series) or not isinstance(groups, pd.Series):
        raise ValueError("y and groups must be indexed pandas Series.")
    if not isinstance(oof, pd.DataFrame):
        raise ValueError("oof must be an indexed pandas DataFrame.")
    if not y.index.is_unique or not oof.index.is_unique or not groups.index.is_unique:
        raise ValueError("y, oof, and groups must have unique row indices.")
    if not y.index.equals(oof.index) or not y.index.equals(groups.index):
        raise ValueError("y, oof, and groups indices must be aligned in the same order.")
    if oof.shape[1] != 3 or not oof.columns.is_unique:
        raise ValueError("oof must have exactly three uniquely named prediction columns.")
    target = _numeric_vector(y, "y")
    _target_weights(target)
    try:
        prediction = oof.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("oof must contain numeric predictions.") from exc
    if not np.isfinite(prediction).all():
        raise ValueError("oof must contain only finite predictions.")
    if groups.isna().any():
        raise ValueError("groups must not contain missing values.")
    try:
        codes, labels = pd.factorize(groups, sort=False)
    except TypeError as exc:
        raise ValueError("groups must contain hashable group identifiers.") from exc
    return target, prediction, codes, labels


def _group_partitions(codes, labels, repeats, n_splits, seed):
    if not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    if not isinstance(repeats, Integral) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if not isinstance(n_splits, Integral) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    if len(labels) < n_splits:
        raise ValueError("The number of distinct groups must be at least n_splits.")
    rng = np.random.default_rng(seed)
    partitions = []
    for repeat in range(repeats):
        # Split group identifiers, never rows. Unequal group sizes are allowed.
        for fold, heldout in enumerate(np.array_split(rng.permutation(len(labels)), n_splits)):
            validation = np.isin(codes, heldout)
            training = ~validation
            partitions.append((training, validation, {
                "repeat": repeat + 1,
                "fold": fold + 1,
                "train_groups": tuple(labels[np.unique(codes[training])]),
                "validation_groups": tuple(labels[np.unique(codes[validation])]),
                "train_rows": int(training.sum()),
                "validation_rows": int(validation.sum()),
                "train_group_count": int(np.unique(codes[training]).size),
                "validation_group_count": int(np.unique(codes[validation]).size),
            }))
    return partitions


def _grid_scores(y, prediction, grid):
    # Do not standardize model predictions: relative scales affect blend signs.
    blended = prediction @ grid.T
    agreement = (blended >= 0) == (y[:, None] >= 0)
    # Use the same accumulation order for every column. A BLAS dot can round
    # identical agreement columns differently and accidentally defeat ties.
    scores = np.sum(_target_weights(y)[:, None] * agreement, axis=0)
    return np.clip(scores, 0, 1)


def learn_common_weights(
    y: pd.Series,
    oof: pd.DataFrame,
    groups: pd.Series,
    repeats: int = 4,
    n_splits: int = 5,
    seed: int = 42,
    step: float = 0.05,
    epsilon: float = 0.01,
) -> CommonWeightsResult:
    """Select a weight vector within epsilon of every training-split optimum.

    The default uses 4 repeats x 5 group folds, giving 20 stability views. In
    each view, the best score and candidate regrets use only the four training
    folds. Common means every such regret is <= epsilon (absolute score units).
    The generic demo tolerance is 0.01, not a competition configuration.

    Rank feasible candidates by full OOF score descending, maximum regret
    ascending, mean regret ascending, then weights lexicographically ascending.
    Full OOF labels therefore participate in selection. selection_score and the
    per-split validation diagnostics are NOT untouched performance estimates.
    An empty common set raises; the requested epsilon is never relaxed.

    Candidate weight_0/1/2 correspond to oof.columns in their given order;
    the same mapping is stored in each table's attrs['model_names'].
    """
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon must be a finite nonnegative scalar.") from exc
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite nonnegative scalar.")
    target, prediction, codes, labels = _validate_inputs(y, oof, groups)
    grid = simplex_grid(step)
    partitions = _group_partitions(codes, labels, repeats, n_splits, seed)
    scores = np.asarray([
        _grid_scores(target[training], prediction[training], grid)
        for training, _, _ in partitions
    ])
    best = scores.max(axis=1)
    regrets = np.maximum(0.0, best[:, None] - scores)
    max_regret = regrets.max(axis=0)
    mean_regret = regrets.mean(axis=0)
    minimum = float(max_regret.min())
    common = max_regret <= epsilon + _TOLERANCE
    if not common.any():
        raise ValueError(
            f"No common grid weights at epsilon={epsilon:.12g}; "
            f"minimum feasible epsilon={minimum:.12g}. "
            "The requested tolerance was not relaxed."
        )
    full_scores = _grid_scores(target, prediction, grid)
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0], mean_regret,
                        max_regret, -full_scores))
    selected = int(next(index for index in order if common[index]))
    candidates = pd.DataFrame(grid, columns=_WEIGHT_COLUMNS).assign(
        full_oof=full_scores, max_regret=max_regret,
        mean_regret=mean_regret, common=common,
    )
    candidates = candidates.iloc[order].reset_index(drop=True)
    diagnostics = pd.DataFrame([
        {**metadata,
         "grid_train_best": float(best[split]),
         "selected_train_score": float(scores[split, selected]),
         "selected_regret": float(regrets[split, selected]),
         "selected_validation_score": weighted_directional_accuracy(
             target[validation], prediction[validation] @ grid[selected]),
         }
        for split, (_, validation, metadata) in enumerate(partitions)
    ])
    for table in (candidates, diagnostics):
        table.attrs["model_names"] = list(oof.columns)
    weights = pd.Series(grid[selected], index=oof.columns, name="weight")
    result = CommonWeightsResult(
        weights=weights,
        candidates=candidates,
        diagnostics=diagnostics,
        epsilon=epsilon,
        min_feasible_epsilon=minimum,
        selection_score=float(full_scores[selected]),
        config={"repeats": int(repeats), "n_splits": int(n_splits), "seed": int(seed),
                "step": float(step), "epsilon": epsilon},
        input_fingerprint=indexed_fingerprint(y, oof, groups),
        source_manifests=dict(oof.attrs.get("source_manifests", {})),
        weights_fingerprint=indexed_fingerprint(weights),
        selector_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    return replace(result, integrity_fingerprint=selection_fingerprint(result))


def cross_fitted_weight_score(
    y: pd.Series,
    oof: pd.DataFrame,
    groups: pd.Series,
    seed: int = 42,
    n_splits: int = 5,
    step: float = 0.05,
) -> pd.DataFrame:
    """Fit grid weights on train groups and score the held-out groups once.

    Each held-out target is excluded from that split's weight learning, including
    tie breaking. Ties use weights in ascending lexical order. Return one row
    per fold, with selected weight_0/1/2, train_score and validation_score. The
    attrs['pooled_score'] value scores all held-out blend predictions together.

    This is a blender-only diagnostic using shared base-model OOF predictions,
    NOT an end-to-end nested estimate: base OOF construction and earlier model,
    feature and hyperparameter selection may still introduce selection bias.
    """
    target, prediction, codes, labels = _validate_inputs(y, oof, groups)
    grid = simplex_grid(step)
    partitions = _group_partitions(codes, labels, 1, n_splits, seed)
    heldout_predictions = np.empty_like(target)
    records = []
    for training, validation, metadata in partitions:
        train_scores = _grid_scores(target[training], prediction[training], grid)
        selected = int(np.argmax(train_scores))  # Grid already has lexical order.
        weights = grid[selected]
        heldout_predictions[validation] = prediction[validation] @ weights
        records.append({
            **metadata,
            **dict(zip(_WEIGHT_COLUMNS, weights)),
            "train_score": float(train_scores[selected]),
            "validation_score": weighted_directional_accuracy(
                target[validation], heldout_predictions[validation]),
        })
    diagnostics = pd.DataFrame(records)
    diagnostics.attrs["pooled_score"] = weighted_directional_accuracy(target, heldout_predictions)
    diagnostics.attrs["model_names"] = list(oof.columns)
    diagnostics.attrs["scope"] = "blender-only cross-fit on shared base OOF; not end-to-end nested CV"
    return diagnostics
