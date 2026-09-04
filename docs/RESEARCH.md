# Research design and limitations

## Separate signal discovery from engineering disclosure

The research problem concerns noisy cross-sectional return reconstruction. Different model classes have different inductive biases: regularized linear fits, boosted trees, and randomized tree ensembles need not make the same prediction errors. That motivates testing a blend; it does not guarantee improvement.

The proprietary feature layer is intentionally absent. The independent synthetic panel provides small numerical covariates and grouped observations solely to make the public code executable. The provider interface hides the complete private transformation process: not only implementation details, but also private feature names, matrix values, fitted transformations, production dimensions, and cache contents.

## Base-model validation

The three public learners share the same date-grouped five-fold split. Each training call receives training labels only; the held-out labels are used by the outer evaluator for scoring, not by the feature-provider call. Ridge imputers and scalers are fitted within model training partitions. The tree models accept missing numerical features directly.

A private provider would need to honor the same contract and independently establish its own leakage controls. Merely implementing an interface does not prove that its internals are safe. Any training features learned from responses require appropriate cross-fitting or other isolation; this repository supplies no such private implementation.

GroupKFold prevents one day from appearing in both sides of a fold. The [organizer explicitly states](https://challengedata.ens.fr/challenges/44) that the day identifiers are randomized and anonymized, with no continuity or link between dates; the official train/test split is random by day with no overlap. Holding out whole day groups follows that experimental unit. Neither sorting these IDs nor dividing their numerical range yields a genuine chronological holdout. A forward-in-time claim would require genuine timestamps or a separately supplied chronological evaluation; neither is provided by this public demo.

## Two-stage Ridge

The first stage pools observations within a generic asset group. The second stage fits target-specific residuals. In the public example both stages use ordinary training-only regression pipelines. This demonstrates pooling and residual modeling, not every numerical optimization or feature treatment in the private competition implementation.

The small regularization search first evaluates a coarse grid, then checks a local grid with an absolute 0.01 spacing. The best point is the best **evaluated** point. Neither the grid nor the score guarantees a global optimum or an unbiased estimate.

The separate public `WeightedRidgePath` solver demonstrates decomposition reuse on already supplied finite design matrices. It solves `sum_i q_i * (y_i - b - x_i @ beta)^2 + alpha * ||beta||^2`, where `q_i` are training sample weights, not the competition's evaluation weights. After weighted centering, one thin SVD of the weighted design gives the shrinkage factors `s / (s^2 + alpha)` for the whole regularization path. Its internal rescaling of sample weights is compensated in alpha so the original objective is preserved. Standardization, when requested, changes the coordinates in which coefficients are penalized. Notebook 01 and the numerical tests compare the result against separate scikit-learn fits.

This solver is a standalone numerical component, not the production feature transform and not a claim of a measured runtime speedup for the whole pipeline. The main public two-stage search retains the transparent estimator-based implementation.

## Tree-model optimization

Both tree families use small public Optuna search spaces. These are independent demonstration settings. Trial history is persisted locally in SQLite; the same group split is reused to compare candidates. Returning to the same OOF scores repeatedly induces selection bias, even though each base prediction was initially out of fold.

## Common-weight selection

Let the rows of `P` be aligned OOF predictions, with columns corresponding to the three models. We evaluate a finite simplex grid of weights `w`, with non-negative entries summing to one, using `P @ w` on its original scale.

For each of four seeded repetitions, split unique days into five groups. For each held-out group, score candidate weights on the other four groups. If `b_j` is the best evaluated score for training partition `j`, candidate `w` is common when:

$$\max_j \{b_j - S_j(w)\} \leq \epsilon.$$

The algorithm finds the highest full-OOF score among the common candidates, then applies deterministic tie-breaks. It reports the minimum epsilon needed for a nonempty intersection on the evaluated grid. If the requested epsilon is too small, it fails explicitly; it does not silently increase tolerance.

"Common" means approximately optimal on all twenty training partitions relative to this grid. It does not mean that twenty mathematically exact optima coincide. A coarser or wider grid can change the intersection. Tolerance, grid resolution, and subsequent full-OOF selection are research choices, not free validation.

## What the scores do not establish

- The final common weight is selected with full OOF labels. Its full-OOF score is a selection score.
- Averaging a fixed selected weight's held-out partition scores cannot turn that score into independent evidence: every observation is reused across repetitions.
- A separate diagnostic learns weights using four partitions and scores the fifth. This checks the second-stage weight-learning rule conditional on existing OOF predictions. It is still not end-to-end nested model selection: base models and their hyperparameters were not refitted inside those outer partitions.
- The historical private pipeline was not uniformly fold-local in all feature preprocessing. This public refactor's interface must not retroactively be presented as proof of a strictly nested historical experiment.
- No transaction costs, turnover, capacity, portfolio constraints, or trading PnL are modeled. A good competition score is not proof of profitable alpha.

## Next research controls

For a stronger empirical claim, freeze the full specification before opening a new untouched set of day groups, quantify group-level uncertainty, compare to simple baselines, and evaluate ablations under a controlled search budget. Genuine temporal testing additionally requires actual time ordering, which cannot be recovered by sorting anonymized IDs. These are proposed controls, not experiments claimed to have been run in this public repository.
