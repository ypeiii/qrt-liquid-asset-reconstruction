# Reproducibility

## Scope

The four notebooks and CLI run entirely on synthetic data. They do not load original competition inputs or any concealed factor cache. The public data generator, seed, model choices, and demo search spaces are visible and independently defined. The case study's aggregate competition statistics were checked separately against private artifacts; the public demo does not regenerate that table or the leaderboard score.

The local development check uses Python 3.14.0 on Windows with NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, LightGBM 4.6.0, Optuna 4.9.0, nbformat 5.10.4, nbclient 0.11.0, and pytest 9.1.1. Dependency ranges in `pyproject.toml` are broader for installation; they are not a bitwise environment lock. Linux CI is configured separately and must actually run after publication before its outcome is claimed.

## Running notebooks

Open the repository root with JupyterLab after installation. Run notebooks 01, 02, and 03 in any order, followed by 04. The three model notebooks use exactly the same synthetic dataset and folds. Their small searches are enabled by default. The CLI without `--search` uses fixed demonstration parameters for a faster smoke test.

Each model saves OOF predictions, full-refit test predictions, and a manifest. The blend verifies the synthetic input fingerprint, row ordering, prediction finiteness, and file hashes before reading the artifacts. Data or artifact changes cannot be silently accepted as the same run.

Fresh notebook executions can be tested with:

```powershell
python tools/run_notebooks.py
```

Execution occurs in a temporary directory. Source notebook outputs remain empty. The test executes the Optuna search paths and full-refit prediction paths, but only at demo scale.

## Optuna resumption

The initial demo target is three valid trials. Completing that batch causes the next invocation to add two more. An unfinished batch resumes toward its stored target. Trial parameters and results survive through SQLite, and a fingerprint rejects incompatible data/configuration reuse. A custom feature provider must supply an explicit nonempty `cache_key` identifying its configuration; its author is responsible for changing that key when its implementation or settings change.

The sampler and estimator seeds stay fixed; the sampler seed does not increase with the trial count. SQLite trial history alone does not checkpoint the entire sampler's random-number state. Restarting a seeded sampler can therefore produce repeated suggestions and is not guaranteed to match a single uninterrupted run bit for bit.

Only one process should optimize a given local study. A process crash may leave a RUNNING trial; the public example fails safely on that state and requires deliberate recovery. Do not delete another process's running trial or remove a database while it is in use. The demo does not claim multi-worker recovery or exact resumption midway through a fitted estimator.

## Re-running and artifacts

Generated files are deliberately excluded from the release. Normal reruns replace that model's local demo prediction files. Optuna databases persist unless a new output directory is selected. To conduct a fresh experiment, use a new `--output-dir` instead of destroying old state.

The weighted sign score has plateaus. Multiple weights may produce the same signs and score. A deterministic tie-break chooses one representative; it does not make that point uniquely optimal.
