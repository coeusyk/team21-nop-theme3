# Task List

## Priority Legend
- **P0** — must do now, blocks everything else
- **P1** — important after P0, core research work
- **P2** — useful stretch work after core is stable

---

## P0 — Setup and Specification

### T1. Verify repo skeleton
Owner: Both\
Output: All folders and empty files as specified in `PROJECT_ARCHITECTURE.md`, `uv sync` passes, VS Code Remote WSL opens cleanly.

### T2. Freeze mathematical specification
Owner: Yash Karecha\
Output: Equation sheet covering:

- LASSO objective: $\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$
- Gradient: $\nabla f(\beta) = -\frac{1}{n}X^\top(y - X\beta)$
- Lipschitz constant: $L = \lambda_{\max}(X^\top X)/n$, step size $\alpha < 1/L$
- Soft-thresholding: $S_\tau(z) = \operatorname{sign}(z)\max(|z| - \tau, 0)$
- Weighted soft-thresholding: $\beta_j^{k+1} = \operatorname{sign}(z_j)\max(|z_j| - \alpha\lambda w_j^{(k)}, 0)$
- IRL1 weight update: $w_j^{(k)} = 1/(|\beta_j^{(k)}| + \epsilon)^\gamma$
- FISTA momentum: $t_{k+1} = (1 + \sqrt{1 + 4t_k^2})/2$

Also write one short positioning note: IRL1 (Candès et al., 2008) is the foundation; this project is a novel empirical application to correlated tabular regression with ISTA/FISTA integration.

### T3. Build preprocessing pipeline
Owner: Tenzin Kunga\
Output:
- `src/data/load_data.py` — raw CSV loader
- `src/data/preprocess.py` — missing values, one-hot encoding, standardization
- `src/data/split.py` — reproducible train/val/test split using seed

Acceptance: One function call produces $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}, y_{\text{train}}, y_{\text{val}}, y_{\text{test}}$ with feature names preserved.

---

## P1 — Core Research Tasks

### T4. Implement proximal utilities
Owner: Yash Karecha\
Files: `src/optim/objective.py`, `src/optim/prox_ops.py`, `src/optim/stopping.py`\
Output:
- `compute_loss(X, y, beta, lam, w)` — returns weighted objective value
- `soft_threshold(z, tau)` — plain $S_\tau(z)$
- `weighted_soft_threshold(z, alpha, lam, w)` — per-coordinate $\beta_j = \operatorname{sign}(z_j)\max(|z_j| - \alpha\lambda w_j, 0)$
- `has_converged(beta_prev, beta_curr, tol)` — relative change stopping criterion

Acceptance: Unit tests pass for edge cases — zero input, all-zero weights, max threshold.

### T5. Implement ISTA solver
Owner: Yash Karecha\
File: `src/optim/ista.py`\
Output: `ista_solve(X, y, lam, w, alpha, max_iter, tol)` → returns `(beta, objective_trace, sparsity_trace, runtime)`\
Acceptance: On fixed-weight plain LASSO ($w_j = 1$), objective decreases monotonically. Solution matches scikit-learn `Lasso` within tolerance on a small synthetic problem.

### T6. Implement FISTA solver
Owner: Yash Karecha\
File: `src/optim/fista.py`\
Output: `fista_solve(X, y, lam, w, alpha, max_iter, tol)` → same signature as ISTA\
Acceptance: Converges to same solution as ISTA in fewer iterations on fixed-weight problems. Momentum uses $t_{k+1} = (1 + \sqrt{1 + 4t_k^2})/2$.

### T7. Implement reweight engine
Owner: Yash Karecha\
File: `src/optim/reweight.py`\
Output:
- `compute_weights(beta, gamma, eps)` — returns weight vector $w_j = 1/(|\beta_j| + \epsilon)^\gamma$
- `clip_weights(w, w_max)` — prevents extreme weights from destabilizing solver

Acceptance: At $\beta = 0$, returns $w_j = 1/\epsilon^\gamma$ for all $j$. Weights decrease as $|\beta_j|$ increases.

### T8. Implement Ridge baseline
Owner: Tenzin Kunga\
File: `src/models/ridge.py`\
Output: Ridge using `sklearn.linear_model.RidgeCV`; saves metrics and selected coefficients.

### T9. Implement LASSO baseline
Owner: Tenzin Kunga\
File: `src/models/lasso.py`\
Output: Standard LASSO using `sklearn.linear_model.LassoCV`; saves metrics and sparsity.

### T10. Implement static adaptive LASSO baseline
Owner: Tenzin Kunga\
File: `src/models/adaptive_lasso.py`\
Output: Run Ridge to get $\hat\beta_\text{ridge}$; compute fixed weights $w_j = 1/(|\hat\beta_j| + \epsilon)^\gamma$; run ISTA/FISTA with those fixed weights; save metrics.

### T11. Implement dynamic reweighted LASSO
Owner: Yash Karecha\
File: `src/models/dynamic_reweighted_lasso.py`\
Output: Outer loop over IRL1 weight updates; calls ISTA or FISTA as inner solver per outer iteration; logs weight evolution, sparsity, and objective per outer step.

---

## P1 — Evaluation Tasks

### T12. Run hyperparameter search
Owner: Tenzin Kunga\
File: `src/experiments/cross_validate.py`\
Search over:
- $\lambda \in [10^{-4},\; 1]$ (log scale)
- $\gamma \in \{0.5,\; 1.0,\; 2.0\}$
- $\epsilon \in \{10^{-3},\; 10^{-4}\}$
- `max_outer_iter` $\in \{5,\; 10,\; 20\}$

Output: Best configs saved to `outputs/logs/best_configs.yaml`; all validation MSE values to CSV.

### T13. Run full experiment comparison
Owner: Tenzin Kunga\
File: `src/experiments/run_baselines.py`, `src/experiments/run_dynamic.py`\
Output: Comparison table with MSE, MAE, $\|\hat\beta\|_0$ (non-zeros), runtime, iterations for all five methods.

### T14. Support stability analysis
Owner: Both\
File: `src/metrics/stability_metrics.py`\
Output: For each method, compute Jaccard index of selected feature sets across 5 seeds or folds:\
$J(A, B) = |A \cap B| / |A \cup B|$\
Acceptance: Stability table included in final outputs.

### T15. Fairness audit
Owner: Both\
Confirm these two comparisons are kept separate in the paper:
1. **Accuracy comparison**: all methods including sklearn baselines, on test-set MSE/MAE.
2. **Optimizer efficiency comparison**: only custom ISTA/FISTA variants, on matched iterations and wall-clock time.

No claim that a custom research prototype "beats" scikit-learn's production solver unless the data explicitly supports it.

---

## P1 — Visualization Tasks

### T16. Convergence plots
Owner: Tenzin Kunga\
File: `src/visualization/convergence_plots.py`\
Output: Objective value vs iteration for ISTA, FISTA, dynamic-ISTA, dynamic-FISTA on same axes. Sparsity ($\|\beta^k\|_0$) vs iteration subplot.

### T17. Sparsity–error tradeoff plot
Owner: Tenzin Kunga\
File: `src/visualization/tradeoff_plots.py`\
Output: Validation MSE vs number of non-zero features as $\lambda$ varies, for all sparse methods.

### T18. Coefficient path plot
Owner: Tenzin Kunga\
File: `src/visualization/coefficient_plots.py`\
Output: Coefficient magnitude vs $\lambda$ for top 20 features; dynamic method vs static LASSO comparison.

---

## P2 — Stretch Tasks

### T19. Synthetic correlated-design experiment
Owner: Yash Karecha\
Generate Toeplitz covariance data with known true support. Compare support recovery accuracy across methods. Demonstrates why $\ell_1$ fails under correlation and why reweighting helps.

### T20. Feature interpretation report
Owner: Both\
List top retained features from dynamic method. Provide domain interpretation for house pricing context. Include in paper's results discussion.

---

## Paper Writing Tasks

### T21. Introduction
Owner: Tenzin Kunga\
Must include: theme motivation, high-dimensional regression challenge, correlated feature problem, why sparse models matter.

### T22. Related Work
Owner: Yash Karecha\
Must cite: LASSO (Tibshirani, 1996), adaptive LASSO (Zou, 2006), IRL1 (Candès et al., 2008), proximal gradient and ISTA/FISTA (Beck & Teboulle, 2009).

### T23. Methodology
Owner: Yash Karecha\
Must include: all equations from T2, algorithm pseudocode box, subdifferential connection, initialization and pruning mechanism explanation, FISTA acceleration, honest convergence scope statement.

### T24. Experiments section
Owner: Tenzin Kunga\
Must include: dataset description, preprocessing protocol, all baselines, metrics, hyperparameter tuning strategy.

### T25. Results and Discussion
Owner: Both\
Must include: best comparison table, convergence figure, sparsity–error tradeoff figure, what worked, what failed, why.

### T26. Limitations section
Owner: Both\
Must explicitly state:
- Dynamic weights change the outer objective each iteration; no global convergence theorem is claimed.
- Efficiency claims depend on fair matched-solver comparison.
- Results may be sensitive to preprocessing and regularization scale.
- No oracle property proof is provided for the dynamic variant.

### T27. Final compliance pass
Owner: Both\
Checklist:
- [ ] Every table and figure in the paper comes from `outputs/`.
- [ ] Paper length is 6–12 pages.
- [ ] Strengths and limitations are explicitly discussed.
- [ ] Plagiarism similarity is within the required threshold.
- [ ] All equations render correctly in the submitted document format.
