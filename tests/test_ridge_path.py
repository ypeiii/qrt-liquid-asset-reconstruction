"""Independent numerical checks, using only freshly generated toy matrices."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from quant_portfolio.ridge_path import WeightedRidgePath


def numerical_example(seed=13):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(37, 6)) * np.arange(1, 7) + 2
    X_eval = rng.normal(size=(11, 6)) * np.arange(1, 7) - 1
    y = 3 + X @ rng.normal(size=6) + rng.normal(size=37)
    weight = rng.uniform(0.2, 3, size=37)
    weight[:3] = 0
    return X, y, X_eval, weight


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_path_matches_independent_sklearn_fits(standardize, weighted):
    X, y, X_eval, sample_weight = numerical_example()
    weight = sample_weight if weighted else None
    alphas = np.array([10, 0.01, 1, 0.01, 1000])
    fitted = WeightedRidgePath(standardize=standardize).fit(X, y, weight)
    actual = fitted.predict(X_eval, alphas)
    if standardize:
        scaler = StandardScaler().fit(X, sample_weight=weight)
        X, X_eval = scaler.transform(X), scaler.transform(X_eval)
    expected = np.column_stack([
        Ridge(alpha=alpha, solver="svd").fit(X, y, sample_weight=weight).predict(X_eval)
        for alpha in alphas
    ])
    assert actual.shape == (len(X_eval), len(alphas))
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(actual[:, 1], actual[:, 3])


@pytest.mark.parametrize("standardize", [False, True])
def test_path_handles_collinearity_constant_features_and_wide_design(standardize):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(7, 15))
    X[:, 1] = X[:, 0]
    X[:, 2] = 7
    X_eval = X[:3].copy()
    y = rng.normal(size=7) + 2
    alphas = [0.0001, 1, 1e4]
    result = WeightedRidgePath(standardize).fit(X, y)
    transformed = (X - result.x_mean_) / result.x_scale_
    transformed_eval = (X_eval - result.x_mean_) / result.x_scale_
    expected = np.column_stack([
        Ridge(alpha=alpha, solver="svd").fit(transformed, y).predict(transformed_eval)
        for alpha in alphas
    ])
    np.testing.assert_allclose(result.predict(X_eval, alphas), expected, rtol=1e-9, atol=1e-9)
    assert result.x_scale_[2] == 1


def test_intercept_is_unpenalized_even_with_constant_only_design():
    X = np.ones((5, 3)) * 7
    y = np.array([0, 2, 4, 8, 9.0])
    weights = np.array([1, 2, 0, 1, 4.0])
    path = WeightedRidgePath(standardize=True).fit(X, y, weights)
    expected = np.full((2, 3), np.average(y, weights=weights))
    np.testing.assert_allclose(path.predict(X[:2], [0.001, 1, 1e10]), expected)


def test_one_training_svd_is_reused_across_alpha_batches(monkeypatch):
    X, y, X_eval, weights = numerical_example()
    original = np.linalg.svd
    calls = []

    def counted_svd(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted_svd)
    path = WeightedRidgePath().fit(X, y, weights)
    first = path.predict(X_eval, [0.1, 1])
    second = path.predict(X_eval, [10, 0.1])
    assert len(calls) == 1
    np.testing.assert_array_equal(first[:, 0], second[:, 1])


def test_evaluation_data_does_not_refit_centering_or_scaling():
    X, y, X_eval, weights = numerical_example()
    path = WeightedRidgePath(standardize=True).fit(X, y, weights)
    before_mean, before_scale = path.x_mean_.copy(), path.x_scale_.copy()
    isolated = path.predict(X_eval[:1], [0.1, 10])
    with_other_rows = path.predict(np.vstack([X_eval[:1], X_eval[1:] * 1000]), [0.1, 10])
    np.testing.assert_allclose(isolated, with_other_rows[:1], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(path.x_mean_, before_mean)
    np.testing.assert_array_equal(path.x_scale_, before_scale)


def test_integer_weights_equal_explicit_row_replication():
    X, y, X_eval, _ = numerical_example()
    weights = np.arange(len(y)) % 4
    copies = np.repeat(np.arange(len(y)), weights)
    weighted = WeightedRidgePath(True).fit(X, y, weights).predict(X_eval, [0.1, 1, 10])
    repeated = WeightedRidgePath(True).fit(X[copies], y[copies]).predict(X_eval, [0.1, 1, 10])
    np.testing.assert_allclose(weighted, repeated, rtol=1e-10, atol=1e-10)


def test_rescaling_training_weights_has_the_expected_alpha_effect():
    X, y, X_eval, weights = numerical_example()
    alphas = np.array([0.1, 1, 10])
    multiplied = WeightedRidgePath().fit(X, y, 7 * weights).predict(X_eval, alphas)
    original = WeightedRidgePath().fit(X, y, weights).predict(X_eval, alphas / 7)
    np.testing.assert_allclose(multiplied, original, rtol=1e-11, atol=1e-11)


def test_zero_weight_rows_cannot_influence_fit():
    X, y, X_eval, weights = numerical_example()
    reference = WeightedRidgePath(True).fit(X, y, weights).predict(X_eval, [0.1, 1])
    X[weights == 0] += 1e6
    y[weights == 0] -= 1e8
    changed = WeightedRidgePath(True).fit(X, y, weights).predict(X_eval, [0.1, 1])
    np.testing.assert_array_equal(reference, changed)


@pytest.mark.parametrize("X,y,weight", [
    ([], [], None), ([1, 2], [1, 2], None), ([[1], [2]], [1], None),
    ([[np.nan]], [1], None), ([[1]], [np.inf], None),
    ([[1], [2]], [1, 2], [1]), ([[1], [2]], [1, 2], [-1, 2]),
    ([[1], [2]], [1, 2], [0, 0]), ([[1]], [1], [np.nan]),
])
def test_fit_rejects_invalid_inputs(X, y, weight):
    with pytest.raises(ValueError):
        WeightedRidgePath().fit(X, y, weight)


@pytest.mark.parametrize("alphas", [[], [0], [-1], [np.nan], [np.inf], [[1]], 1])
def test_predict_rejects_invalid_alphas(alphas):
    path = WeightedRidgePath().fit([[1], [2]], [2, 3])
    with pytest.raises(ValueError):
        path.predict([[3]], alphas)


def test_predict_requires_fit_and_matching_finite_features():
    path = WeightedRidgePath()
    with pytest.raises(RuntimeError, match="fit"):
        path.predict([[1]], [1])
    path.fit([[1], [2]], [2, 3])
    with pytest.raises(ValueError, match="features"):
        path.predict([[1, 2]], [1])
    with pytest.raises(ValueError, match="finite"):
        path.predict([[np.inf]], [1])
    assert path.predict(np.empty((0, 1)), [1, 2]).shape == (0, 2)


def test_failed_refit_does_not_leave_previous_model_usable():
    path = WeightedRidgePath().fit([[1], [2]], [2, 3])
    with pytest.raises(ValueError):
        path.fit([[np.nan]], [2])
    with pytest.raises(RuntimeError, match="fit"):
        path.predict([[1]], [1])
