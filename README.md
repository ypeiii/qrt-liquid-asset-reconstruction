# Reconstruction of Liquid Asset Performance

Quantitative machine-learning research by **yang.pei** for [QRT / ENS ChallengeData #44](https://challengedata.ens.fr/challenges/44).

**Public leaderboard rank: 5. Displayed public score: 0.7511. Username: `yang.pei`.**
The [author-provided leaderboard screenshot](docs/evidence/public-leaderboard.png) shows the challenge, public ranking, username, and score. The author reports 484 entries; that total is not visible in the screenshot. This is a recorded public-board result, not a final placement or a live ranking check. See [result provenance](docs/RESULTS.md).

This project studies how to combine a pooled linear model with target-specific tree models for noisy cross-asset prediction. The public implementation exposes the modeling, numerical methods, validation, parameter search, and ensemble logic. **Proprietary feature construction and feature-bearing artifacts remain private.** An independent synthetic dataset makes the workflow runnable without distributing that research input; it does not reproduce the competition score.

## Research decisions

| Question | Decision and reasoning | Code / discussion |
| --- | --- | --- |
| What should be held out together? | Entire anonymized days, matching the organizer's split unit; day IDs have no chronological meaning | [Research design](docs/RESEARCH.md) |
| Why three model families? | Pool common structure with Ridge, fit target-specific residuals, and compare tree models with different inductive biases | [Model implementation](src/quant_portfolio/models.py) |
| How can repeated linear fits share computation? | Fit one weighted design decomposition and reuse its spectral shrinkage across regularization strengths, checked against independent fits | [Numerical solver](src/quant_portfolio/ridge_path.py) |
| How are parameters compared? | Shared group folds, a coarse-to-fine Ridge grid, and persistent Optuna studies with fixed seeds and context checks | [Search implementation](src/quant_portfolio/search.py) |
| Why not simply maximize one weight-search score? | Inspect the intersection of near-optimal weight sets across twenty resampled training partitions, then report its selection limitations | [Blend implementation](src/quant_portfolio/blend.py) |
| How can a silent blend error be caught? | Check ordered IDs, input fingerprints, finite predictions, and prediction-file hashes before combining models | [Artifact implementation](src/quant_portfolio/artifacts.py) |

## Review the project

Start with the [competition case study](docs/COMPETITION_CASE_STUDY.md), then the [code review guide](docs/CODE_REVIEW_GUIDE.md). The four notebooks provide an executable walkthrough:

1. [Two-stage Ridge](notebooks/01_two_stage_ridge.ipynb)
2. [LightGBM](notebooks/02_lightgbm.ipynb)
3. [ExtraTrees](notebooks/03_extra_trees.ipynb)
4. [Common-weight ensemble](notebooks/04_ensemble.ipynb)

The [test suite](tests) and [CI configuration](.github/workflows/ci.yml) exercise numerical correctness, group separation, held-out-label isolation, search resumption, and artifact alignment. Actual local test results are recorded in [verification](docs/VERIFICATION.md); remote CI is not claimed until it has run.

## Problem and metric

The competition reconstructs liquid-asset return directions from same-day illiquid-asset returns. The organizer anonymizes and randomizes day IDs and splits train/test by disjoint days. Sorting those IDs is therefore not a chronological validation scheme. [Official task and data description](https://challengedata.ens.fr/challenges/44).

The score is absolute-return-weighted directional accuracy:

$$ S(y,\hat y) = \frac{\sum_i |y_i|\,\mathbf{1}[(y_i\geq0)=(\hat y_i\geq0)]}{\sum_i |y_i|}. $$

The models predict a continuous response. A non-negative raw prediction becomes `+1`, otherwise `-1`. The blend operates on raw predictions, without standard-deviation normalization. This weighted metric must not be described as ordinary unweighted classification accuracy.

## Public workflow

```text
Independent synthetic panel
  -> date-grouped folds -> fold-scoped feature interface
  -> pooled two-stage Ridge / LightGBM / ExtraTrees
  -> aligned out-of-fold predictions
  -> 4 repetitions x 5 weight-selection partitions
  -> intersection of epsilon-near-optimal candidate weights
  -> full-data refits -> raw predictions -> synthetic submission-style file
```

| Component | Demonstrated implementation |
| --- | --- |
| Ridge | Pool-level prediction plus target-level residual regression; fold-local imputation and scaling; coarse search followed by a local 0.01 grid |
| LightGBM | Per-target regression; native missing-value handling; Optuna search |
| ExtraTrees | Per-target regression; native missing-value handling; Optuna search |
| Experiment state | SQLite trial persistence, fixed random seeds, bounded resume batches, data/configuration fingerprints |
| Ensemble | Non-negative weights summing to one; day-grouped stability analysis; explicit infeasibility and deterministic tie-breaking |
| Prediction files | Unique row IDs, exact index alignment, finite-value checks, and SHA-256 verification |

The executable search spaces and synthetic features are demonstration choices, **not** production settings. The case study separately identifies facts checked against the private research record. Feature construction is omitted altogether rather than disguised through renamed columns.

## Run locally

Python 3.12 or newer is required. From the repository root, on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m quant_portfolio.demo --model all
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` instead.
Add `--search` to exercise the small hyperparameter searches. The command never downloads competition data, reads a private project, or submits predictions to a website.

For an interactive walkthrough:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Run notebooks `01` through `04` in order. The first three create local synthetic OOF and test artifacts; the fourth reads and validates them before blending. Its weights and scores are synthetic and will not equal the competition weights or score.

Generated outputs live in the ignored `artifacts/` directory. Synthetic test labels are deliberately not returned by the data generator, so the demo reports OOF diagnostics only, not a test-performance claim. See [reproducibility notes](docs/REPRODUCIBILITY.md) for resume semantics and environment limits.

## Validation limits matter

Group folds keep a day intact. The challenge does not disclose usable chronology, so they do not establish forward-in-time generalization. Hyperparameters are selected on OOF scores, and the final common weight is selected on full OOF; those reported scores are selection diagnostics, not untouched estimates. The repeated weight partitions reuse the same OOF predictions and are not 20 new model-training runs.

The public demo is a newly factored engineering example, not a verbatim release of the competition notebooks. Its fold-local preprocessing contract must not be read as proof that every preprocessing step in the historical competition pipeline was fold-local. The [research note](docs/RESEARCH.md) separates historical limitations from the public interface.

## Disclosure and release boundary

Only the public training/validation framework and independent synthetic demonstration are included. No private feature provider is supplied, and simply placing the original competition files beside these notebooks will not reproduce the competition run.

The original leaderboard screenshot is included as reviewed evidence. Review the challenge-specific sharing rules and [the release checklist](docs/PUBLISHING.md) before redistributing or adding material. This project is independent and is not affiliated with or endorsed by QRT, ENS, or a prospective employer.

## License

The author's original code and documentation are licensed under the [MIT License](LICENSE). The leaderboard screenshot and third-party content are excluded from that grant; see [license scope and third-party notices](THIRD_PARTY_NOTICES.md). Competition data and proprietary feature construction are not distributed or licensed by this public release.
