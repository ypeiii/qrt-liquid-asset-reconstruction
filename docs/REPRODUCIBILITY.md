# Reproducibility

## Scope

The four notebooks and CLI run entirely on synthetic data. They do not load original competition inputs or any concealed factor cache. The public data generator, seed, model choices, and demo search spaces are visible and independently defined. The case study's aggregate competition statistics were checked separately against private artifacts; the public demo does not regenerate that table or the leaderboard score.

The local development check uses Python 3.14.0 on Windows with NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, LightGBM 4.6.0, Optuna 4.9.0, nbformat 5.10.4, nbclient 0.11.0, and pytest 9.1.1. Dependency ranges in `pyproject.toml` are broader for installation; they are not a bitwise environment lock. Linux CI is configured separately and must actually run after publication before its outcome is claimed.

## Running notebooks

Open the repository root with JupyterLab after installation. Run notebooks 01, 02, and 03 in any order, followed by 04. The three model notebooks use exactly the same synthetic dataset and folds. Their small searches are enabled by default. The CLI without `--search` uses fixed demonstration parameters for a faster smoke test.

`model_oof` and `refit_predict` attach their actual generation context to each prediction Series. The context records the panel, model, parameters, provider identity, seed, relevant implementation hashes, and runtime versions; OOF predictions also carry the exact fold protocol. `save_model_artifacts` rejects missing, modified, or inconsistent prediction metadata instead of trusting a separately supplied description.

Every successful save creates an immutable `<model>/runs/<run_id>/` directory containing the OOF and full-refit files. Only after both payloads are complete does one atomic replacement make `<model>/manifest.json` point to that generation. Prior runs are retained. The blend loader checks all three active manifests, their file hashes, exact row order, finite predictions, and shared protocol before returning a prediction stack.

Fresh notebook executions can be tested with:

```powershell
python tools/run_notebooks.py
```

Execution occurs in a temporary directory. Source notebook outputs remain empty. The test executes the Optuna search paths and full-refit prediction paths, but only at demo scale.

## Optuna resumption

The initial demo target is three valid trials. Completing that batch causes the next invocation to add two more. An unfinished batch resumes toward its stored target. Trial parameters and results survive through SQLite, and a fingerprint rejects incompatible data/configuration reuse. A custom feature provider must supply an explicit nonempty `cache_key` identifying its configuration; its author is responsible for changing that key when its implementation or settings change. Invalid model names, trial budgets, folds, and seeds are rejected before the storage directory or database is created.

The sampler and estimator seeds stay fixed; the sampler seed does not increase with the trial count. SQLite trial history alone does not checkpoint the entire sampler's random-number state. Restarting a seeded sampler can therefore produce repeated suggestions and is not guaranteed to match a single uninterrupted run bit for bit.

Only one process should optimize a given local study. A process crash may leave a RUNNING trial; the public example fails safely on that state and requires deliberate recovery. Do not delete another process's running trial or remove a database while it is in use. The demo does not claim multi-worker recovery or exact resumption midway through a fitted estimator.

## Re-running and artifacts

Generated files are deliberately excluded from the release. A normal model rerun adds a new immutable generation and atomically changes only the active manifest; it does not overwrite or remove previous runs. An interrupted generation cannot become active before its manifest switch. This protocol assumes one writer per model/output directory and is not presented as a universal power-loss durability guarantee.

Artifact schema 1 is not upgraded in place because it lacks the model-generated context required by schema 2. Choose a fresh `--output-dir`, or set `PORTFOLIO_ARTIFACT_DIR` to a fresh directory for the notebooks, then rerun all three model notebooks followed by the ensemble notebook. Leave the older directory and its SQLite studies intact. Optuna continuation behavior itself is unchanged; choosing a fresh artifact directory starts separate demo studies without deleting prior history.

## Selection-to-export provenance

`learn_common_weights` returns the selected weights together with the exact step, epsilon, repeats, folds, seed, input fingerprint, and source-model manifest hashes used to select them. Notebook 04 passes this object directly to `run_ensemble(selected=selected)`. Export does not silently repeat the search. Any explicitly supplied export control must match `selected.config`; changed weights, OOF inputs, or active model manifests are rejected and require a fresh load and selection.

Each export creates an immutable `ensemble/runs/<run_id>/` bundle and atomically switches the active ensemble manifest after completion. That manifest contains the complete selection configuration, snapshots and SHA-256 digests of all three source manifests, and a SHA-256 digest for every exported file. `verify_ensemble_artifacts` follows those pinned source snapshots rather than whichever model manifests are currently active. Consequently, a previously exported ensemble remains verifiable after a newer model generation becomes active, provided its referenced immutable runs are retained.

The weighted sign score has plateaus. Multiple weights may produce the same signs and score. A deterministic tie-break chooses one representative; it does not make that point uniquely optimal.
