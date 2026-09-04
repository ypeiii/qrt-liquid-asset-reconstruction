# Competition case study

## Problem framing

QRT's [Reconstruction of Liquid Asset Performance](https://challengedata.ens.fr/challenges/44) asks for liquid-asset return directions given contemporaneous illiquid-asset returns. The business motivation is to understand relationships between less tradable and more tradable instruments. The supplied task is supervised reconstruction, not an implemented trading strategy.

The unit of validation matters: the organizer splits by disjoint anonymized days, and the day numbers do not encode time order. Grouping by day avoids splitting a common cross-section across training and validation. Sorting the anonymized IDs would not supply the missing chronology.

## Modeling choices

The research pipeline combines a pooled two-stage Ridge model, per-target LightGBM, and per-target ExtraTrees. The first Ridge stage estimates shared group-level structure; its second stage models target-specific residuals. The tree families provide different nonlinear approximations to the same prediction problem.

Models produce raw continuous predictions, while evaluation depends on their signs and the magnitudes of the true returns. Parameter selection therefore uses the competition's weighted directional score rather than assuming a lower regression loss necessarily implies a better challenge score. The ensemble combines raw predictions without standard-deviation normalization.

Private feature construction is a separate layer. No feature formulas, transformations, semantic names, production dimensions, matrix values, or learned feature state are disclosed by this case study.

## A verified local snapshot

The following values were checked against the private run record. All six model OOF/test file hashes match the stored snapshot, and the five selected-ensemble artifact hashes match its manifest. Diagnostic row IDs, labels, and OOF predictions also agree with the corresponding model files. Only aggregate results are disclosed here.

| Model or ensemble | Full-OOF weighted directional score |
| --- | ---: |
| Two-stage Ridge | 0.74184730 |
| LightGBM | 0.74875791 |
| ExtraTrees | 0.74698485 |
| Equal-weight three-model ensemble | 0.74923899 |
| Selected common-weight ensemble | 0.75017490 |

The selected weights are **Ridge 0.15910, LightGBM 0.34723, ExtraTrees 0.49367**. On this snapshot the selected blend exceeds equal weighting by **0.09359 percentage points** and the best single model by **0.14170 percentage points**. These are in-sample-to-selection OOF comparisons, not estimates of a causal or independently significant improvement.

The separately supplied [leaderboard screenshot](evidence/public-leaderboard.png) shows **public rank 5, public score 0.7511**, under `yang.pei`. The author reports a total field size of 484, which is not visible in the image. The local audit has not established which uploaded submission received that online score. Do not identify the local snapshot above as the score-0.7511 submission without a matching platform record. See [result provenance](RESULTS.md).

## Why a weaker standalone model can still matter

The paired error analysis below uses the same aligned local OOF snapshot. Each cell is a share of **total absolute-return weight across all observations**, not a conditional error probability and not an unweighted percentage of rows.

| First / second model | Both correct | Both wrong | Only first correct | Only second correct |
| --- | ---: | ---: | ---: | ---: |
| Ridge / LightGBM | 69.9337% | 20.8731% | 4.2511% | 4.9421% |
| Ridge / ExtraTrees | 69.4394% | 20.5561% | 4.7454% | 5.2591% |
| LightGBM / ExtraTrees | 71.7437% | 22.1694% | 3.1321% | 2.9548% |

Rows sum to 100% up to rounding. Across all three models, 68.0964% of absolute-return weight is classified correctly by every model, 19.2613% incorrectly by every model, and 12.6423% has mixed correctness.

This shows why standalone rank alone is insufficient for ensemble selection: even Ridge has observations on which it is correct and the stronger standalone model is not. It motivates testing complementary predictions. It does not provide an oracle to identify those observations without labels, or prove an improvement on unseen data.

## Weight selection and stability

The private weight search uses fixed OOF predictions and four repetitions of five group partitions. In each of the twenty views, candidate weights are evaluated on four partitions. A weight is eligible when its regret against the best evaluated candidate is within the selected tolerance on every view.

| Snapshot diagnostic | Value |
| --- | ---: |
| Epsilon, in absolute score units | 0.00031 |
| Eligible candidates on the final evaluated grid | 169 |
| Eligible candidates tied at the highest full-OOF score | 141 |
| Selected weight's maximum training-partition regret | 0.00030548 |

The 141 tied candidates are a score plateau, not 141 independently corroborated discoveries. A deterministic tie-break selects one representative. Reporting a weight to five decimal places does not imply that it has five-decimal statistical precision. The candidate grids were refined locally; neither the intersection nor the resulting weight is claimed to be a continuous global optimum.

The public notebook demonstrates the same common-weight principle on a much smaller independent grid. Its selected weights and scores are synthetic and should differ from the table above.

## Computation and experiment control

The original research used matrix-factorization reuse to evaluate Ridge regularization paths. The public [Ridge path solver](../src/quant_portfolio/ridge_path.py) isolates that general numerical idea and tests it against scikit-learn on independent matrices. It is a reviewed public refactor, not a copy of the private feature pipeline or every optimization in the production implementation.

The public tree searches persist trial history and batch targets in SQLite. They reject changes in dataset fingerprint, feature-provider version, folds, seed, search-space version, or runtime context. The prediction loader independently checks file hashes and ordered IDs before blending. Those controls prevent comparisons and submissions from silently mixing incompatible experiment outputs.

## Boundaries of the evidence

The model hyperparameters and final weight were selected using OOF labels. The repeated weight partitions are a stability diagnostic, not twenty new model-training experiments or an untouched holdout. Some historical feature preprocessing was not fitted separately within every outer fold, so the original result is not described as end-to-end nested validation.

No claim is made about transaction costs, portfolio performance, or an investable alpha. A stronger next experiment would freeze the full pipeline before accessing a new untouched collection of day groups and evaluate uncertainty at the group level. That experiment has not been claimed as completed here.
