"""Small synthetic checks for blending, group isolation, and selection scope."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.blend import (
    cross_fitted_weight_score,
    learn_common_weights,
    simplex_grid,
    weighted_directional_accuracy,
)


def synthetic_oof(seed=7):
    rng = np.random.default_rng(seed)
    index = pd.Index([f"row_{i}" for i in range(40)], name="row")
    y = pd.Series(rng.normal(size=40), index=index)
    oof = pd.DataFrame(rng.normal(size=(40, 3)), index=index,
                       columns=["model_a", "model_b", "model_c"])
    groups = pd.Series(np.repeat(np.arange(10), 4), index=index)
    return y, oof, groups


def test_weighted_directional_accuracy_manual_and_zero_sign():
    assert weighted_directional_accuracy([-2, 0, 3, -5], [-1, -0.5, -1, -1]) == pytest.approx(0.7)
    assert weighted_directional_accuracy([-1, 1], [0, 0]) == pytest.approx(0.5)
    assert weighted_directional_accuracy([1e308, -1e308], [1, -1]) == pytest.approx(1)


@pytest.mark.parametrize("y,p", [
    ([], []), ([1], [1, 2]), ([[1]], [[1]]), ([0, 0], [1, -1]),
    ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf]),
])
def test_metric_rejects_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        weighted_directional_accuracy(y, p)


def test_metric_rejects_series_misalignment():
    with pytest.raises(ValueError, match="aligned"):
        weighted_directional_accuracy(pd.Series([1, -1], index=[1, 2]),
                                      pd.Series([1, -1], index=[2, 1]))


def test_simplex_is_complete_nonnegative_and_lexically_ordered():
    grid = simplex_grid()
    assert grid.shape == (231, 3)
    assert np.all(grid >= 0)
    np.testing.assert_allclose(grid.sum(axis=1), 1)
    np.testing.assert_allclose(grid * 20, np.round(grid * 20))
    assert grid.tolist() == sorted(grid.tolist())
    np.testing.assert_array_equal(simplex_grid(1), [[0, 0, 1], [0, 1, 0], [1, 0, 0]])


@pytest.mark.parametrize("step", [0, -0.1, 2, 0.03, np.nan, np.inf])
def test_simplex_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        simplex_grid(step)


def test_common_weights_twenty_group_isolated_views_and_determinism():
    y, oof, groups = synthetic_oof()
    result = learn_common_weights(y, oof, groups, step=0.25, epsilon=1)
    repeated = learn_common_weights(y, oof, groups, step=0.25, epsilon=1)
    assert len(result.diagnostics) == 20
    assert list(result.weights.index) == list(oof.columns)
    assert result.weights.sum() == pytest.approx(1)
    assert result.candidates.common.all()
    assert result.epsilon == 1
    assert result.min_feasible_epsilon == result.candidates.max_regret.min()
    assert result.selection_score == pytest.approx(
        weighted_directional_accuracy(y, oof @ result.weights)
    )
    for _, repeat in result.diagnostics.groupby("repeat"):
        heldout = []
        for row in repeat.itertuples():
            assert set(row.train_groups).isdisjoint(row.validation_groups)
            assert row.train_rows + row.validation_rows == len(y)
            assert row.train_group_count + row.validation_group_count == groups.nunique()
            assert row.selected_regret <= result.epsilon + 1e-12
            heldout.extend(row.validation_groups)
        assert sorted(heldout) == sorted(groups.unique())
    pd.testing.assert_series_equal(result.weights, repeated.weights)
    pd.testing.assert_frame_equal(result.candidates, repeated.candidates)
    pd.testing.assert_frame_equal(result.diagnostics, repeated.diagnostics)


def test_no_common_set_raises_and_reports_minimum_without_relaxation():
    y = pd.Series(np.ones(6))
    groups = pd.Series(np.repeat(np.arange(3), 2))
    prediction = -np.ones((6, 3))
    prediction[np.arange(6), groups.to_numpy()] = 1
    oof = pd.DataFrame(prediction, columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="minimum feasible epsilon=0.5"):
        learn_common_weights(y, oof, groups, repeats=1, n_splits=3, step=1, epsilon=0)
    result = learn_common_weights(y, oof, groups, repeats=1, n_splits=3, step=1, epsilon=0.5)
    assert result.min_feasible_epsilon == pytest.approx(0.5)


def test_candidate_regret_and_order_match_independent_calculation():
    y, oof, groups = synthetic_oof()
    result = learn_common_weights(y, oof, groups, repeats=1, step=0.5, epsilon=1)
    candidates = result.candidates
    train_scores = []
    for split in result.diagnostics.itertuples():
        training = groups.isin(split.train_groups)
        train_scores.append([
            weighted_directional_accuracy(
                y.loc[training], oof.loc[training] @ np.asarray(weights)
            )
            for weights in candidates[["weight_0", "weight_1", "weight_2"]].to_numpy()
        ])
    train_scores = np.asarray(train_scores)
    regrets = train_scores.max(axis=1)[:, None] - train_scores
    np.testing.assert_allclose(candidates.max_regret, regrets.max(axis=0), atol=1e-12)
    np.testing.assert_allclose(candidates.mean_regret, regrets.mean(axis=0), atol=1e-12)
    ordered = candidates.sort_values(
        ["full_oof", "max_regret", "mean_regret", "weight_0", "weight_1", "weight_2"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(candidates, ordered)
    np.testing.assert_array_equal(
        result.weights.to_numpy(),
        candidates.loc[candidates.common, ["weight_0", "weight_1", "weight_2"]].iloc[0].to_numpy(),
    )


def test_lexical_tie_break_is_deterministic():
    y, oof, groups = synthetic_oof()
    oof[:] = 1
    result = learn_common_weights(y, oof, groups, step=0.5, epsilon=0)
    np.testing.assert_array_equal(result.weights.to_numpy(), [0, 0, 1])


@pytest.mark.parametrize("function", [learn_common_weights, cross_fitted_weight_score])
def test_blend_rejects_unaligned_or_nonfinite_input(function):
    y, oof, groups = synthetic_oof()
    with pytest.raises(ValueError, match="aligned"):
        function(y, oof.iloc[::-1], groups)
    with pytest.raises(ValueError, match="aligned"):
        function(y, oof, groups.iloc[::-1])
    oof.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        function(y, oof, groups)


def test_heldout_label_perturbation_cannot_change_that_splits_weight_learning():
    y, oof, groups = synthetic_oof()
    first = cross_fitted_weight_score(y, oof, groups, step=0.25)
    heldout_groups = first.iloc[0].validation_groups
    changed_y = y.copy()
    changed_y.loc[groups.isin(heldout_groups)] *= -17
    changed = cross_fitted_weight_score(changed_y, oof, groups, step=0.25)
    columns = ["weight_0", "weight_1", "weight_2", "train_score"]
    pd.testing.assert_series_equal(first.iloc[0][columns], changed.iloc[0][columns])
    assert len(first) == 5
    assert "not end-to-end nested" in first.attrs["scope"]
    assert 0 <= first.attrs["pooled_score"] <= 1
    for row in first.itertuples():
        assert set(row.train_groups).isdisjoint(row.validation_groups)


def test_predictions_are_blended_on_raw_scale():
    y = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    oof = pd.DataFrame({"a": [10, -10] * 3, "b": [-1, 1] * 3, "c": [-1, 1] * 3})
    groups = pd.Series(np.repeat(np.arange(3), 2))
    result = learn_common_weights(y, oof, groups, n_splits=3, step=0.25, epsilon=0)
    candidate = result.candidates.query("weight_0 == 0.25 and weight_1 == 0.25")
    assert candidate.iloc[0].full_oof == pytest.approx(1)
