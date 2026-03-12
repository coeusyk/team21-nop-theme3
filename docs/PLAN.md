# Project Plan

## Goal
Build a complete machine learning pipeline for Theme 3 that formulates the optimization objective, implements a custom dynamic proximal-gradient optimizer, evaluates it against standard baselines, and produces evidence for the final 6–12 page technical paper.

## Ground Rules
- The project and the paper are one linked deliverable; the paper must only report experiments actually run in the codebase.
- The implementation must be modular, reproducible, and executable.
- The final report must include methodology, equations, graphs, tables, convergence behavior, strengths, and limitations.

## Strategy
1. Use reweighted $\ell_1$ minimization as the literature-backed foundation (Candès et al., 2008).
2. Implement it inside a proximal-gradient framework with ISTA and FISTA.
3. Evaluate honestly on correlated tabular regression.
4. Claim a solid empirical and implementation contribution — not a fake theoretical breakthrough.

## Mathematical Foundations (must be locked before coding)

### LASSO

```math
\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1
```

### Weighted LASSO (dynamic objective at outer iteration $k$)

```math
\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} w_j^{(k)} |\beta_j|
```

### IRL1 weight update

```math
w_j^{(k)} = \frac{1}{\left(|\beta_j^{(k)}| + \epsilon\right)^\gamma}
```

### Proximal gradient step

```math
\beta^{k+1} = \operatorname{prox}_{\alpha \lambda w^{(k)}}(\beta^k - \alpha \nabla f(\beta^k))
```

### Weighted soft-thresholding (proximal operator of weighted $\ell_1$)

```math
\beta_j^{k+1} = \operatorname{sign}(z_j^k)\max\!\left(|z_j^k| - \alpha \lambda w_j^{(k)},\; 0\right)
```

### FISTA convergence rate
ISTA achieves $O(1/k)$ convergence. FISTA accelerates this to $O(1/k^2)$ for fixed composite objectives by adding the momentum update:

```math
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, \quad y^{k+1} = \beta^k + \frac{t_k - 1}{t_{k+1}}(\beta^k - \beta^{k-1})
```

## Milestones

### Milestone 1 — Math and literature lock
Deliverables:
- Clean hand-derivation of soft-thresholding from proximal definition
- Positioning note: IRL1 as foundation, project contribution as empirical application

Exit criteria: No equation in the paper is hand-wavy; no false novelty claim exists.

### Milestone 2 — Data pipeline
Deliverables:
- Dataset loader, imputer, encoder, scaler, split script
- Saved processed matrices with preserved feature names

Exit criteria: One command produces reproducible train/val/test matrices.

### Milestone 3 — Baselines
Deliverables:
- Ridge, standard LASSO, static adaptive LASSO
- Metrics table and CV loop

Exit criteria: Sparsity and MSE reported; CV is working.

### Milestone 4 — Custom solvers
Deliverables:
- ISTA implementation
- FISTA implementation
- Weighted proximal operator
- Logging and stopping criteria

Exit criteria: Objective decreases monotonically in ISTA fixed-weight mode on toy data.

### Milestone 5 — Dynamic reweighting
Deliverables:
- Outer IRL1 reweight loop
- Weight update module with $\gamma$, $\epsilon$, clipping, max outer iterations
- Full experiment runner

Exit criteria: Dynamic mode converges stably on synthetic data; runtime and sparsity traces saved.

### Milestone 6 — Final experiments
Deliverables:
- Full comparison table (MSE, MAE, sparsity, runtime, iterations)
- Convergence plots
- Sparsity–error tradeoff plots
- Support stability analysis

Exit criteria: Results are repeatable across seeds; figures are paper-ready.

### Milestone 7 — Paper writing
Deliverables:
- 6–12 page draft with equations, algorithm box, result discussion, limitations
- Plagiarism-safe original writing

Exit criteria: Every figure/table comes from repo outputs; no unsupported claim; similarity below required threshold.

## Experimental Protocol

### Dataset
House Prices: Advanced Regression Techniques (Kaggle). Strongly correlated predictors stress-test feature-selection stability.

### Splits
- 60% train / 20% validation / 20% test
- Or: 5-fold CV on train+validation, final retrain and test evaluation

### Models
| Method | Type | Notes |
|---|---|---|
| Ridge | Baseline | Dense, no selection |
| Standard LASSO | Baseline | Unweighted $\ell_1$ |
| Static adaptive LASSO | Baseline | Fixed weights from Ridge |
| Dynamic IRL1 + ISTA | Proposed | Dynamic weights, $O(1/k)$ inner |
| Dynamic IRL1 + FISTA | Proposed | Dynamic weights, $O(1/k^2)$ inner |

### Metrics
- MSE and MAE on validation and test sets
- Number of non-zero coefficients
- Support stability: Jaccard index of selected features across folds/seeds
- Runtime (wall-clock) and iteration count to convergence under matched stopping criteria

## Risk Register

| Risk | Response |
|---|---|
| Dynamic method has worse MSE than Ridge | Tune $\lambda$, $\gamma$, $\epsilon$; report sparsity–accuracy tradeoff honestly |
| Dynamic method is slower than plain LASSO | Reweighting adds overhead; only claim efficiency if matched-so
