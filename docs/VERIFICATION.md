# Local verification record

The following checks were actually run while preparing this public release, on the Windows / Python 3.14 environment described in [reproducibility notes](REPRODUCIBILITY.md):

| Check | Observed result |
| --- | --- |
| `python -m pytest -q` | 86 tests passed |
| `python tools/run_notebooks.py` | All four notebooks completed; 21 code cells executed in total |
| Weighted Ridge path | 30 numerical tests, included above; independent scikit-learn fits, rank deficiency, training-weight semantics, and decomposition reuse |
| CLI, all three models and ensemble | Completed; generated and read back a synthetic submission-style file |
| Release checks | Explicit allowlist, reviewed screenshot SHA-256, empty notebook outputs, metadata checks, and exposure-pattern checks passed |
| Disclosure review | No private feature construction, feature matrices, or real row-level prediction files found in the prepared public source; aggregate competition statistics are intentionally included |
| Local competition record | Six model artifact hashes and five selected-ensemble artifact hashes matched their private manifest; diagnostic IDs, labels, and predictions were aligned |

The notebook smoke run used temporary synthetic artifacts and left source notebook outputs empty. The CLI generated only synthetic artifacts, which are excluded from the release package. Notebook code is unchanged by the screenshot-evidence update; the full unit-test suite was rerun after that update. The 33-file release allowlist includes the public numerical component, aggregate case study, and original reviewed screenshot, not the private source artifacts used to check the case study.

These checks establish behavior of the public example on the tested environment. They are not verification of the leaderboard result, a guarantee against every possible future disclosure, a remote GitHub CI result, or evidence of trading profitability. Repeat the checks after changing the code or dependencies.
