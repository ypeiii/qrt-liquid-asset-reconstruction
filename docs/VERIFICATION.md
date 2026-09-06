# Local verification record

This record combines release-preparation checks with the experiment-provenance and export repair recheck on September 6, 2026 (Asia/Shanghai), on the Windows / Python 3.14 environment described in [reproducibility notes](REPRODUCIBILITY.md). The complete test suite, all four notebooks, a custom-grid CLI workflow, and the release checker were rerun after this repair; historical checks are identified separately below.

| Check | Observed result |
| --- | --- |
| `python -m pytest -q` | 142 tests passed, including 44 new experiment-provenance and search-preflight cases added to the previous 98-test suite |
| `python tools/run_notebooks.py` | All four notebooks completed; 21 code cells executed in total |
| Weighted Ridge path | 30 numerical tests, included above; independent scikit-learn fits, rank deficiency, training-weight semantics, and decomposition reuse |
| CLI, all three models and ensemble | `--model all --step 0.1 --epsilon 1` completed in a temporary directory; the saved ensemble passed `verify_ensemble_artifacts`, retained the requested grid step, and pinned all three model generations |
| Exact selected-result export | Passed: the notebook exports its existing selection without a second search; conflicting controls, stale inputs, and modified selection contents are rejected |
| Generation integrity and interrupted writes | Passed: actual generation context and file hashes are checked; an interrupted publication leaves the previous complete generation active; a saved ensemble remains verifiable after base models are rerun |
| Optuna argument preflight | Passed: invalid folds, seeds, or budgets are rejected before creating storage or changing an existing study; existing database bytes remain unchanged |
| Release checks | All 38 allowlisted files passed; reviewed screenshot SHA-256, empty notebook outputs, metadata checks, and exposure-pattern checks passed |
| Disclosure review | No private feature construction, feature matrices, or real row-level prediction files found in the prepared public source; aggregate competition statistics are intentionally included |
| Local competition record | Checked during release preparation: six model artifact hashes and five selected-ensemble artifact hashes matched their private manifest; diagnostic IDs, labels, and predictions were aligned |

The notebook smoke run used temporary synthetic artifacts and left source notebook outputs empty. The CLI generated only synthetic artifacts, which are excluded from the release package. The 38-file release allowlist includes the public numerical component, aggregate case study, original reviewed screenshot, MIT license, and third-party notices, not the private source artifacts used to check the case study.

The earlier publication repair restored `.gitignore` and `.github/workflows/ci.yml` and added packaging guards. This repair changes public prediction metadata, artifact storage, ensemble export, search argument validation, and the notebook workflow. It does not change the underlying learning algorithms, the MIT license text, or the screenshot. Schema 1 demo artifacts must be regenerated for all three models in a fresh output directory; existing private artifacts and Optuna databases need not be deleted. For commit-specific Linux CI results, consult [GitHub Actions](https://github.com/ypeiii/qrt-liquid-asset-reconstruction/actions); the Windows results here are not a substitute for a successful remote run.

These checks establish behavior of the public example on the tested environment. They are not verification of the leaderboard result, a guarantee against every possible future disclosure, a remote GitHub CI result, or evidence of trading profitability. Repeat the checks after changing the code or dependencies.
