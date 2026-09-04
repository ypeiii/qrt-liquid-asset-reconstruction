"""A reusable weighted Ridge regularization path, independent of any features.

This public numerical refactor follows the spectral-reuse idea in the private
competition workflow. It is not the production model: it accepts an arbitrary
finite design matrix and contains no feature construction or production inputs.
One training-only SVD supports every subsequent regularization choice.
"""

import numpy as np


def _finite_array(values, name, dimensions):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if result.ndim != dimensions or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {dimensions}-dimensional array.")
    return result


class WeightedRidgePath:
    """Solve weighted squared loss plus alpha times the squared coefficient norm.

    The intercept is never penalized. With ``standardize=True``, coefficients
    are penalized in coordinates standardized using the *training* weighted
    population standard deviation. Constant columns use scale one. Thus this
    option changes the regularization geometry, not merely the implementation.

    ``sample_weight`` is not a scoring weight: it enters the training loss.
    Multiplying it by c is equivalent to dividing alpha by c. For every positive
    alpha, predictions are well-defined even for a rank-deficient design.
    The class does not choose alpha or provide an independent performance score.
    """

    def __init__(self, standardize=False):
        if not isinstance(standardize, (bool, np.bool_)):
            raise ValueError("standardize must be True or False.")
        self.standardize = bool(standardize)
        self._is_fitted = False

    def fit(self, X, y, sample_weight=None):
        """Fit centering/scaling and one thin SVD using training observations."""
        self._is_fitted = False
        features = _finite_array(X, "X", 2)
        target = _finite_array(y, "y", 1)
        if not features.shape[0] or not features.shape[1]:
            raise ValueError("X must contain at least one row and one feature.")
        if len(target) != len(features):
            raise ValueError("X and y must have the same number of rows.")
        weight = (np.ones(len(features)) if sample_weight is None
                  else _finite_array(sample_weight, "sample_weight", 1))
        if len(weight) != len(features) or np.any(weight < 0) or not np.any(weight > 0):
            raise ValueError("sample_weight must match X and be nonnegative with positive mass.")

        # Exclude zero-weight rows and rescale weights for numerical safety.
        # The same scale is applied to alpha at prediction time, preserving the
        # original (unnormalized) weighted Ridge objective exactly in arithmetic.
        positive = weight > 0
        features, target, weight = features[positive], target[positive], weight[positive]
        weight_scale = float(weight.max())
        relative_weight = weight / weight_scale
        mean_weight = relative_weight / relative_weight.sum()
        x_mean = mean_weight @ features
        y_mean = float(mean_weight @ target)
        constant = np.all(features == features[0], axis=0)
        x_mean[constant] = features[0, constant]
        with np.errstate(over="ignore", invalid="ignore"):
            centered = features - x_mean
            centered_y = target - y_mean
        if not np.isfinite(centered).all() or not np.isfinite(centered_y).all():
            raise ValueError("Centered training values exceed floating-point range.")
        x_scale = np.ones(features.shape[1])
        if self.standardize:
            # Scale before squaring, so variance does not overflow unnecessarily.
            magnitude = np.max(np.abs(centered), axis=0)
            safe_magnitude = np.where(magnitude > 0, magnitude, 1)
            variance_unit = mean_weight @ np.square(centered / safe_magnitude)
            x_scale = safe_magnitude * np.sqrt(variance_unit)
            x_scale[x_scale == 0] = 1
        root_weight = np.sqrt(relative_weight)
        matrix = (centered / x_scale) * root_weight[:, None]
        response = centered_y * root_weight
        if not np.isfinite(matrix).all() or not np.isfinite(response).all():
            raise ValueError("Weighted training values exceed floating-point range.")

        # SVD works directly on X, avoiding the condition-number squaring in X'X.
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        projected_target = left.T @ response
        if not np.isfinite(projected_target).all():
            raise ValueError("Projected training response exceeds floating-point range.")
        self.n_features_in_ = features.shape[1]
        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.y_mean_ = y_mean
        self.singular_values_ = singular
        self._right_vectors = right
        self._projected_target = projected_target
        self._weight_scale = weight_scale
        self._is_fitted = True
        return self

    def predict(self, X_eval, alphas):
        """Return shape (n_evaluation_rows, n_alphas), preserving alpha order.

        Repeated alpha values are allowed. No decomposition or preprocessing fit
        occurs here; the evaluation rows cannot change the training statistics.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before predict.")
        features = _finite_array(X_eval, "X_eval", 2)
        penalties = _finite_array(alphas, "alphas", 1)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("X_eval has a different number of features than training X.")
        if not len(penalties) or np.any(penalties <= 0):
            raise ValueError("alphas must be a nonempty vector of strictly positive values.")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            effective_alpha = penalties / self._weight_scale
        if not np.isfinite(effective_alpha).all() or np.any(effective_alpha <= 0):
            raise ValueError("The alpha/sample-weight ratio exceeds floating-point range.")
        singular = self.singular_values_[:, None]
        # s / (s**2 + a), computed without squaring a potentially large s.
        root_alpha = np.sqrt(effective_alpha)[None, :]
        scale = np.maximum(singular, root_alpha)
        relative_s = singular / scale
        shrinkage = (relative_s / (relative_s**2 + (root_alpha / scale)**2)) / scale
        with np.errstate(over="ignore", invalid="ignore"):
            spectral_coefficients = shrinkage * self._projected_target[:, None]
            coefficients = self._right_vectors.T @ spectral_coefficients
            transformed = (features - self.x_mean_) / self.x_scale_
            prediction = transformed @ coefficients + self.y_mean_
        if not np.isfinite(prediction).all():
            raise ValueError("Predictions exceed floating-point range.")
        return prediction
