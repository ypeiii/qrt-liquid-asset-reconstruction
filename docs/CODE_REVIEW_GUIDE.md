# Code review guide

The four notebooks provide a short executable narrative. The modules below expose the implementation decisions without exposing competition feature construction.

## 1. Numerical linear algebra

Read [`ridge_path.py`](../src/quant_portfolio/ridge_path.py) and its [reference tests](../tests/test_ridge_path.py). The regularization path reuses one decomposition of an already supplied design matrix. Weighted centering handles an unpenalized intercept; optional scaling is estimated from training observations only. Tests compare each prediction path with independent scikit-learn Ridge fits and exercise rank-deficient designs.

This is a regression solver. It does not construct or select the private research features. Notebook 01 demonstrates it alongside the ordinary estimator-based two-stage training loop, so numerical equivalence is inspectable without claiming that the demonstration is the exact competition implementation.

## 2. Model evaluation and information boundaries

Read `model_oof` and `_fit_predict` in [`models.py`](../src/quant_portfolio/models.py). Check that a full day is held out together, only training labels reach `FeatureProvider.prepare`, and each output row is predicted once. The synthetic provider learns no factors; the private provider has no distributed implementation.

The [pipeline tests](../tests/test_pipeline.py) exercise disjoint fit/evaluation groups, exact row alignment, all-missing covariates for Ridge, native tree missing-value handling, and full-data refitting.

## 3. Stateful hyperparameter search

Read `run_optuna_search` in [`search.py`](../src/quant_portfolio/search.py). The batch target survives interruption. A completed batch adds a bounded number of trials on the next invocation. The seed remains fixed, but the code explicitly does not promise that a re-created sampler resumes the uninterrupted random-number stream.

Tests cover interrupted-batch accounting, active-trial protection, context mismatches, provider-version mismatches, and real tiny tree-model trials. The public demo does not claim multi-worker recovery or resumption halfway through an estimator fit.

## 4. Non-smooth ensemble selection

Read `learn_common_weights` and `cross_fitted_weight_score` in [`blend.py`](../src/quant_portfolio/blend.py). The score changes when a blended prediction crosses zero. Multiple weight vectors can therefore share a score plateau. Inspect the finite candidate grid, per-partition regrets, explicit empty-intersection error, and deterministic tie-break.

The [blend tests](../tests/test_blend.py) check the twenty grouped views, raw prediction scales, independently reconstructed regrets, and held-out-label perturbations. The second diagnostic cross-fits only the blender on existing OOF predictions; it does not claim end-to-end nested model selection.

## 5. Artifact integrity

Read [`artifacts.py`](../src/quant_portfolio/artifacts.py) and its [tests](../tests/test_artifacts.py). A successful fit is insufficient if predictions are then combined in the wrong row order or from different data versions. The loader requires exact IDs, fingerprints, manifest filenames, hashes, and finite values before matrix multiplication.

The repository-wide [release checker](../tools/release_check.py) separately enforces a publication allowlist and empty notebook outputs. The sole permitted binary is the original leaderboard screenshot, pinned to its manually reviewed SHA-256. That protects the publication boundary, not model validity. Any future feature-provider implementation still needs its own audit.
