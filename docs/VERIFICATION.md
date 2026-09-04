# Local verification record

This record combines release-preparation checks with the publication-repair recheck on September 5, 2026 (Asia/Shanghai), on the Windows / Python 3.14 environment described in [reproducibility notes](REPRODUCIBILITY.md). The complete test suite, all four notebooks, and release checker were rerun after the repair; historical checks are identified separately below.

| Check | Observed result |
| --- | --- |
| `python -m pytest -q` | 98 tests passed, including 12 additional publication-guard regression cases |
| `python tools/run_notebooks.py` | All four notebooks completed; 21 code cells executed in total |
| Weighted Ridge path | 30 numerical tests, included above; independent scikit-learn fits, rank deficiency, training-weight semantics, and decomposition reuse |
| CLI, all three models and ensemble | Passed during release preparation; generated and read back a synthetic submission-style file; runtime model code unchanged by the publication repair |
| Release checks | Explicit allowlist, reviewed screenshot SHA-256, empty notebook outputs, metadata checks, and exposure-pattern checks passed |
| Disclosure review | No private feature construction, feature matrices, or real row-level prediction files found in the prepared public source; aggregate competition statistics are intentionally included |
| Local competition record | Checked during release preparation: six model artifact hashes and five selected-ensemble artifact hashes matched their private manifest; diagnostic IDs, labels, and predictions were aligned |

The notebook smoke run used temporary synthetic artifacts and left source notebook outputs empty. The CLI generated only synthetic artifacts, which are excluded from the release package. The 35-file release allowlist includes the public numerical component, aggregate case study, original reviewed screenshot, MIT license, and third-party notices, not the private source artifacts used to check the case study.

The publication repair restores `.gitignore` and `.github/workflows/ci.yml`, requires the license and critical publishing files in the allowlist, and tests that packaged files exactly match the reviewed inputs. Model code, notebook code, the MIT license text, and the screenshot are unchanged by this repair. For commit-specific Linux CI results, consult [GitHub Actions](https://github.com/ypeiii/qrt-liquid-asset-reconstruction/actions); the Windows results here are not a substitute for a successful remote run.

These checks establish behavior of the public example on the tested environment. They are not verification of the leaderboard result, a guarantee against every possible future disclosure, a remote GitHub CI result, or evidence of trading profitability. Repeat the checks after changing the code or dependencies.
