# Code review guide

The four notebooks provide a short executable narrative. The modules below expose the implementation decisions without exposing competition feature construction.

## 1. Numerical linear algebra

Read [`ridge_path.py`](../src/quant_portfolio/ridge_path.py) and its [reference tests](../tests/test_ridge_path.py). The regularization path reuses one decomposition of an already supplied design matrix. Weighted centering handles an unpenalized intercept; optional scaling is estimated from training observations only. Tests compare each prediction path with independent scikit-learn Ridge fits and exercise rank-deficient designs.

This is a regression solver. It does not construct or select the private research features. Notebook 01 demonstrates it alongside the ordinary estimator-based two-stage training loop, so numerical equivalence is inspectable without claiming that the demonstration is the exact competition implementation.

## 2. Model evaluation and information boundaries

Read `model_oof` and `_fit_predict` in [`models.py`](../src/quant_portfolio/models.py). Check that a full day is held out together, only training labels reach `FeatureProvider.prepare`, and each output row is predicted once. Then inspect [`provenance.py`](../src/quant_portfolio/provenance.py): both OOF and full-refit predictions carry a content-bound generation context, while OOF also records the exact group-fold assignments. The synthetic provider learns no factors; the private provider has no distributed implementation.

The [pipeline tests](../tests/test_pipeline.py) exercise disjoint fit/evaluation groups, exact row alignment, all-missing covariates for Ridge, native tree missing-value handling, and full-data refitting.

## 3. Stateful hyperparameter search

Read `run_optuna_search` in [`search.py`](../src/quant_portfolio/search.py). The batch target survives interruption. A completed batch adds a bounded number of trials on the next invocation. The seed remains fixed, but the code explicitly does not promise that a re-created sampler resumes the uninterrupted random-number stream.

Tests cover invalid arguments before storage creation, interrupted-batch accounting, active-trial protection, context mismatches, provider-version mismatches, and real tiny tree-model trials. The public demo does not claim multi-worker recovery or resumption halfway through an estimator fit.

## 4. Non-smooth ensemble selection

Read `learn_common_weights` and `cross_fitted_weight_score` in [`blend.py`](../src/quant_portfolio/blend.py). The score changes when a blended prediction crosses zero. Multiple weight vectors can therefore share a score plateau. Inspect the finite candidate grid, per-partition regrets, explicit empty-intersection error, and deterministic tie-break. The returned selection also binds its configuration, input fingerprint, source-manifest hashes, and selected weights.

The [blend tests](../tests/test_blend.py) check the twenty grouped views, raw prediction scales, independently reconstructed regrets, and held-out-label perturbations. The second diagnostic cross-fits only the blender on existing OOF predictions; it does not claim end-to-end nested model selection. In [`demo.py`](../src/quant_portfolio/demo.py), verify that an existing selection is exported directly: a conflicting explicit control, changed weight, changed OOF stack, or newly active model manifest requires a new selection rather than a silent re-search.

## 5. Artifact integrity

Read [`artifacts.py`](../src/quant_portfolio/artifacts.py) and its [tests](../tests/test_artifacts.py). A successful fit is insufficient if predictions are then combined in the wrong row order or from different data versions. Schema-2 model artifacts require metadata attached by the actual model calls. Each save writes a new immutable `runs/<run_id>/` generation and atomically switches the active manifest only after every payload is complete.

The ensemble manifest records the complete selection configuration plus snapshots and hashes of the three source manifests. `verify_ensemble_artifacts` follows those pinned snapshots and recomputes both the raw blend and thresholded output. This lets an older ensemble remain verifiable after a newer model run becomes active, as long as the referenced immutable runs remain present. Legacy schema-1 outputs must be regenerated in a fresh output directory; no old Optuna database needs to be deleted.

The repository-wide [release checker](../tools/release_check.py) separately enforces a publication allowlist and empty notebook outputs. The sole permitted binary is the original leaderboard screenshot, pinned to its manually reviewed SHA-256. That protects the publication boundary, not model validity. Any future feature-provider implementation still needs its own audit.
