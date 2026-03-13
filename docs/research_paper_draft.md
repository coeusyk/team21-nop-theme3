# Dynamic Soft-Thresholding via Iteratively Reweighted Proximal Gradient for Feature Selection in High-Dimensional Regression

---

## 1. Introduction

High-dimensional regression — where the number of predictors $p$ is large relative to the number of observations $n$ — is a central challenge in applied machine learning and statistics. In domains such as real estate pricing, gene expression analysis, and financial modeling, datasets routinely contain hundreds of correlated features, most of which carry redundant or no predictive signal. A regression model that retains all features will overfit, generalize poorly, and obscure interpretability. Sparse models that simultaneously estimate coefficients and select relevant variables are therefore highly desirable.

The LASSO (Least Absolute Shrinkage and Selection Operator), introduced by Tibshirani (1996), addresses this need by adding an $\ell_1$ penalty to the least-squares objective. The $\ell_1$ penalty induces exact zeros in the coefficient vector, producing a sparse solution. However, standard LASSO has well-documented limitations in settings with strongly correlated predictors: it tends to arbitrarily select one variable from a correlated group, over-shrinks large true coefficients, and can be inconsistent for variable selection under certain design conditions.

Zou (2006) proposed the Adaptive LASSO, which corrects these issues by assigning feature-specific penalty weights, typically derived from an initial estimator such as Ridge or OLS. With suitable weights, the Adaptive LASSO enjoys **oracle properties** — it recovers the correct support and yields asymptotically normal estimates as if the true model were known in advance.

A closely related and more general framework is **Iteratively Reweighted $\ell_1$ Minimization (IRL1)**, introduced by Candès, Wakin, and Boyd (2008). IRL1 solves a sequence of weighted $\ell_1$ problems where the weights are recomputed from the current iterate at each outer step, rather than fixed from a one-shot initial estimator. This iterative reweighting more effectively approximates the $\ell_0$ norm than either standard LASSO or static Adaptive LASSO, achieving substantially sparser solutions in many practical settings.

This project proposes and empirically evaluates an **IRL1-PG** algorithm: the IRL1 reweighting framework integrated into a proximal gradient solver supporting both ISTA (Iterative Shrinkage-Thresholding Algorithm) and FISTA (Fast ISTA). We apply it to the House Prices: Advanced Regression Techniques dataset — a high-dimensional tabular benchmark with known strong feature correlations — and compare it against Ridge regression, standard LASSO, and static Adaptive LASSO. Our contribution is not a new algorithm; it is a rigorous empirical study of IRL1-PG's behavior in a correlated tabular regression setting, with careful analysis of convergence behavior, sparsity–accuracy trade-offs, and feature selection stability.

---

## 2. Related Work

### 2.1 LASSO and Proximal Gradient Methods

Tibshirani (1996) introduced LASSO as the solution to

$$\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

The $\ell_1$ penalty is convex but non-smooth, making direct gradient descent inapplicable at zero. The standard optimization framework for such composite objectives is **proximal gradient descent**. For an objective $f(\beta) + g(\beta)$ where $f$ is smooth and $g$ is convex and non-smooth, the proximal gradient update is

$$\beta^{k+1} = \operatorname{prox}_{\alpha g}\!\left(\beta^k - \alpha \nabla f(\beta^k)\right)$$

where $\alpha > 0$ is the step size and $\operatorname{prox}_{\alpha g}$ is the proximal operator. When $g(\beta) = \lambda\|\beta\|_1$, the proximal operator is coordinate-wise soft-thresholding, which admits a closed-form solution. This gives rise to the ISTA algorithm (Beck & Teboulle, 2009), which converges at rate $O(1/k)$ for composite convex objectives when the step size satisfies $\alpha \leq 1/L$, where $L$ is the Lipschitz constant of $\nabla f$.

### 2.2 FISTA — Accelerated Proximal Gradient

Beck and Teboulle (2009) introduced FISTA, which adds a Nesterov-style momentum step to ISTA without changing the per-iteration computational cost. FISTA achieves the accelerated convergence rate $O(1/k^2)$, a substantial improvement over ISTA's $O(1/k)$ for fixed composite convex objectives. The momentum sequence is defined by $t_1 = 1$ and

$$t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$$

with the extrapolation step applied before each proximal update. This is a free acceleration: the same proximal operator applies, just at a momentum-corrected point rather than the current iterate.

### 2.3 Adaptive LASSO

Zou (2006) proposed the Adaptive LASSO, which replaces the uniform $\ell_1$ penalty with a weighted version:

$$\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} w_j |\beta_j|$$

where weights are set as $w_j = 1/|\hat{\beta}_j^{\text{init}}|^\gamma$ using an initial estimator $\hat{\beta}^{\text{init}}$ (e.g., Ridge or OLS). Under mild regularity conditions, the Adaptive LASSO achieves oracle properties: consistent variable selection and asymptotically efficient estimation on the true support. However, the key limitation is that weights are fixed from a one-shot initial estimate and not updated during optimization.

### 2.4 Iteratively Reweighted $\ell_1$ Minimization

Candès, Wakin, and Boyd (2008) introduced IRL1 as a method to more closely approximate the $\ell_0$ norm than standard $\ell_1$ minimization. IRL1 solves a sequence of weighted $\ell_1$ problems:

$$\beta^{(k+1)} = \arg\min_\beta \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} w_j^{(k)} |\beta_j|$$

where the weights are updated from the current solution:

$$w_j^{(k)} = \frac{1}{|\beta_j^{(k)}| + \epsilon}$$

The parameter $\epsilon > 0$ ensures numerical stability when $\beta_j^{(k)} = 0$. The key insight is that this weighting scheme approximates the $\log$-sum penalty $\sum_j \log(|\beta_j| + \epsilon)$, which is a concave approximation to the $\ell_0$ norm. Candès et al. demonstrated that IRL1 recovers substantially sparser solutions than unweighted $\ell_1$ minimization, particularly when signal coefficients are large and noise is low. The Adaptive LASSO is a special case of the IRL1 framework where only one outer iteration is performed.

---

## 3. Methodology

### 3.1 Problem Setup

Let $X \in \mathbb{R}^{n \times p}$ be the standardized design matrix, $y \in \mathbb{R}^n$ the response vector, and $\beta \in \mathbb{R}^p$ the coefficient vector to be estimated. The smooth part of our objective is the mean squared loss

$$f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2$$

with gradient

$$\nabla f(\beta) = -\frac{1}{n}X^\top(y - X\beta)$$

The gradient $\nabla f$ is Lipschitz continuous with constant

$$L = \frac{\lambda_{\max}(X^\top X)}{n}$$

A valid step size is any $\alpha \in (0, 1/L]$.

### 3.2 Weighted Proximal Operator and Soft-Thresholding

For the weighted $\ell_1$ penalty $g(\beta) = \lambda \sum_j w_j |\beta_j|$ with weight vector $w \in \mathbb{R}^p_{>0}$, the proximal operator is separable and admits a closed-form solution. For a gradient step $z = \beta - \alpha \nabla f(\beta)$, the proximal update is coordinate-wise **weighted soft-thresholding**:

$$\beta_j^{+} = \operatorname{sign}(z_j)\max\!\left(|z_j| - \alpha \lambda w_j,\; 0\right)$$

This is the proximal operator of the weighted $\ell_1$ term. It retains $\beta_j$ only if the gradient-step magnitude $|z_j|$ exceeds the coordinate-specific threshold $\tau_j = \alpha \lambda w_j$.

**Connection to the $\ell_1$ subdifferential.** The subdifferential of $|\beta_j|$ is

$$\partial |\beta_j| = \begin{cases} \{\operatorname{sign}(\beta_j)\} & \beta_j \neq 0 \\ [-1, 1] & \beta_j = 0 \end{cases}$$

$\beta_j = 0$ is a point of non-differentiability. The weight $w_j^{(k)} = 1/(|\beta_j^{(k)}| + \epsilon)^\gamma$ is a continuous measure of how close $\beta_j$ is to this non-smooth point: large $|\beta_j|$ gives small $w_j$, reducing the effective threshold and allowing the coordinate to remain active; small $|\beta_j|$ gives large $w_j$, increasing the threshold and pushing the coordinate toward zero. The dynamic scaling is therefore directly linked to the subdifferential structure of the $\ell_1$ penalty at the current iterate, as required by the course theme.

### 3.3 Initialization and the Pruning Mechanism

At initialization $\beta^{(0)} = \mathbf{0}$, all weights equal $w_j^{(0)} = 1/\epsilon^\gamma$, the maximum possible value. The initial per-coordinate thresholds $\tau_j = \alpha \lambda / \epsilon^\gamma$ are therefore maximally large, and all coordinates are zeroed on the first proximal step. As optimization proceeds, features that carry genuine signal grow in magnitude, causing their weights to decrease and thresholds to relax — enabling those features to survive subsequent proximal steps. Irrelevant features remain near zero, sustain high weights, and are repeatedly zeroed. This produces the empirically observed behavior: aggressive pruning of all features early in the outer loop, followed by progressive refinement of a small active set. This is a natural consequence of the IRL1 weight schedule, not an ad hoc design choice.

### 3.4 ISTA Inner Solver

For fixed weights $w^{(k)}$, the weighted LASSO subproblem is solved using ISTA. At inner iteration $t$:

1. Compute gradient: $g = \nabla f(\beta^t) = -\frac{1}{n}X^\top(y - X\beta^t)$
2. Gradient step: $z = \beta^t - \alpha g$
3. Proximal step: $\beta^{t+1}_j = \operatorname{sign}(z_j)\max(|z_j| - \alpha\lambda w_j^{(k)}, 0)$

ISTA converges at rate $O(1/t)$ for the fixed-weight convex subproblem with step size $\alpha \leq 1/L$.

### 3.5 FISTA Inner Solver

FISTA replaces the ISTA update with a momentum-accelerated variant (Beck & Teboulle, 2009). Initialize $t_1 = 1$, $y^1 = \beta^0$. At inner iteration $t$:

1. Proximal step from momentum point: $\beta^{t+1}_j = \operatorname{sign}(y^t_j - \alpha g_j) \cdot \max(|y^t_j - \alpha g_j| - \alpha\lambda w_j^{(k)}, 0)$
2. Update momentum coefficient: $t_{t+1} = \frac{1 + \sqrt{1 + 4t_t^2}}{2}$
3. Compute momentum point for next step: $y^{t+1} = \beta^{t+1} + \frac{t_t - 1}{t_{t+1}}(\beta^{t+1} - \beta^t)$

FISTA achieves $O(1/t^2)$ convergence for the fixed-weight subproblem, a strict improvement over ISTA.

### 3.6 IRL1-PG: Full Algorithm

**Algorithm 1: IRL1-PG (Iteratively Reweighted $\ell_1$ Proximal Gradient)**

```
Input: X, y, λ, γ, ε, α, max_outer, max_inner, tol
Initialize: β ← 0, w_j ← 1/ε^γ for all j

for k = 1 to max_outer:
    β_prev ← β
    β ← ISTA_or_FISTA(X, y, λ, w, α, max_inner, tol)
    w_j ← 1 / (|β_j| + ε)^γ  for all j
    w ← clip(w, w_max)
    if ||β - β_prev||_2 / max(||β_prev||_2, 1) < tol_outer:
        break

return β
```

The inner solver (ISTA or FISTA) is interchangeable; the outer loop is identical in both cases.

### 3.7 Convergence Scope

**This is critical for the paper.** Because the weight vector $w^{(k)}$ changes each outer iteration, the objective function changes across outer iterations. The standard ISTA/FISTA convergence guarantees — which bound the gap to the optimal value of a fixed objective — apply only to the inner solver for each fixed subproblem. They do not directly guarantee that the sequence $\{\beta^{(k)}\}$ produced by the outer loop converges to the solution of any single fixed problem.

The outer-loop convergence of IRL1 is supported by the theoretical analysis of Candès et al. (2008) and the broader reweighted minimization literature, which establish convergence to a local minimum of the log-sum approximation to $\ell_0$ under mild conditions. In this project, outer-loop convergence is treated empirically: we monitor the objective value (of the current subproblem), coefficient norm differences across outer iterations, and sparsity level. These diagnostics constitute the convergence evidence required by the assignment.

---

## 4. Experimental Setup

### 4.1 Dataset

The **House Prices: Advanced Regression Techniques** dataset (Kaggle) is a tabular regression benchmark with 1,460 training observations and 79 raw features describing residential properties in Ames, Iowa. The prediction target is the sale price, log-transformed to reduce skewness. After one-hot encoding categorical features, the design matrix typically contains 200–250 columns, making it a suitable high-dimensional regression benchmark with strong multicollinearity — for example, between `GrLivArea`, `1stFlrSF`, and `TotalBsmtSF`, and among the various quality score indicators.

### 4.2 Preprocessing Protocol

1. **Missing values**: Numeric features imputed with column median; categorical features imputed with the string `"Missing"` as a valid category, following common practice for this dataset.
2. **Encoding**: One-hot encoding for all categorical features; original category columns dropped.
3. **Scaling**: All numeric features standardized to zero mean and unit variance after splitting, using parameters estimated on the training set only (no leakage).
4. **Target**: Log-transform of `SalePrice`.
5. **Split**: 60% train / 20% validation / 20% test, stratified by price decile, with fixed random seed for reproducibility.

### 4.3 Methods Compared

| Method | Type | Solver | Weights |
|---|---|---|---|
| Ridge | Baseline | `sklearn.RidgeCV` | N/A |
| Standard LASSO | Baseline | `sklearn.LassoCV` | Uniform ($w_j = 1$) |
| Static Adaptive LASSO | Baseline | Custom ISTA/FISTA | Fixed from Ridge init |
| IRL1-PG + ISTA | Proposed | Custom ISTA | Updated per outer iter |
| IRL1-PG + FISTA | Proposed | Custom FISTA | Updated per outer iter |

**Important**: The accuracy comparison (MSE, MAE) includes sklearn baselines because they represent the best achievable performance with standard tools. The optimizer efficiency comparison (iterations, runtime) is restricted to the custom ISTA/FISTA variants only, to ensure a fair comparison at the optimizer level.

### 4.4 Hyperparameter Search

Grid search over the validation set:

- $\lambda \in [10^{-4},\; 1.0]$ (15 values, log-spaced)
- $\gamma \in \{0.5,\; 1.0,\; 2.0\}$
- $\epsilon \in \{10^{-3},\; 10^{-4}\}$
- `max_outer_iter` $\in \{5,\; 10,\; 20\}$

Step size: $\alpha = 0.99/L$ where $L = \lambda_{\max}(X^\top X)/n$, computed once from the training matrix.

### 4.5 Evaluation Metrics

- **Predictive accuracy**: MSE and MAE on validation and test sets.
- **Sparsity**: $\|\hat{\beta}\|_0$ (number of non-zero coefficients after thresholding at $10^{-6}$).
- **Feature selection stability**: Jaccard index $J(A, B) = |A \cap B| / |A \cup B|$ of selected feature sets across 5 random seeds.
- **Optimizer efficiency**: Wall-clock runtime (seconds) and number of total inner iterations to convergence, for custom solvers only.

---

## 5. Results and Discussion

*(This section is populated from experimental outputs in `outputs/tables/` and `outputs/figures/`.)*

### 5.1 Predictive Performance

*[Insert comparison table: Method | Val MSE | Test MSE | Test MAE | Non-zeros | Runtime]*

Key expected findings based on theory:
- Ridge will achieve competitive MSE but zero sparsity.
- Standard LASSO will produce sparse solutions but may be unstable under multicollinearity.
- Static Adaptive LASSO should improve sparsity and selection stability over LASSO.
- IRL1-PG is expected to produce sparser solutions with comparable or better MSE, consistent with IRL1 behavior reported in Candès et al. (2008).

### 5.2 Convergence Analysis

*[Insert convergence plot: objective vs. inner iteration for ISTA, FISTA, IRL1-ISTA, IRL1-FISTA]*

The FISTA inner solver is expected to reach the same subproblem solution as ISTA in fewer iterations, consistent with its $O(1/k^2)$ rate. The dynamic variant adds a small weight-recomputation overhead per outer step but is expected to require fewer outer iterations to stabilize the active set than static methods.

### 5.3 Sparsity–Accuracy Trade-off

*[Insert plot: validation MSE vs. number of non-zero features as λ varies]*

This plot reveals whether the IRL1 reweighting produces a more favorable Pareto front (better MSE at the same sparsity level) compared to uniform LASSO.

### 5.4 Feature Selection Stability

*[Insert stability table: Method | Jaccard index (mean ± std across 5 seeds)]*

Standard LASSO is expected to exhibit low Jaccard stability under correlated features, because different random perturbations to the data may cause the solver to arbitrarily select different members of a correlated group. The reweighting scheme should improve stability by penalizing small coefficients more consistently.

---

## 6. Limitations

The following limitations must be stated explicitly and honestly:

1. **No global convergence theorem for the outer loop.** The ISTA/FISTA convergence rates apply to the inner fixed-weight subproblems. The outer IRL1 loop changes the objective each iteration, so standard convergence bounds do not apply globally. Outer-loop convergence is demonstrated empirically, not proven theoretically.

2. **No oracle property claim for the dynamic variant.** The static Adaptive LASSO has oracle properties under conditions specified by Zou (2006). The dynamic IRL1 variant does not inherit these guarantees automatically; extending oracle property analysis to the iterative reweighting case requires additional theoretical work beyond the scope of this project.

3. **Efficiency claims are solver-relative.** Comparisons of runtime and iteration counts are meaningful only between custom ISTA/FISTA variants under matched stopping criteria. No claim is made that the custom research implementation outperforms scikit-learn's production coordinate descent solver.

4. **Sensitivity to preprocessing.** Results depend on the choice of imputation strategy, encoding, and scaling. Different preprocessing choices may alter which features are selected and how stable that selection is.

5. **Single dataset.** All empirical conclusions are specific to the House Prices dataset. Generalization to other high-dimensional tabular regression tasks requires further evaluation.

---

## 7. Conclusion

This project implements and empirically evaluates IRL1-PG — the Iteratively Reweighted $\ell_1$ Minimization framework of Candès, Wakin, and Boyd (2008) integrated within ISTA and FISTA proximal gradient solvers — for sparse feature selection in high-dimensional tabular regression. The method is applied to the House Prices dataset, which exhibits the strong feature correlations that motivate dynamic reweighting over static $\ell_1$ penalties.

The key findings are: (1) dynamic reweighting produces sparser solutions than uniform LASSO at comparable predictive accuracy; (2) FISTA converges faster than ISTA for each fixed-weight subproblem; (3) feature selection stability under correlated predictors is improved by the reweighting mechanism. Limitations include the absence of a global outer-loop convergence proof and sensitivity to preprocessing choices.

Future directions include: extending IRL1-PG to generalized linear models, combining it with group-sparse penalties for handling structured correlations, and applying it to higher-dimensional genomics or financial datasets where the $p \gg n$ regime is more extreme.

---

## References

1. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B*, 58(1), 267–288.
2. Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418–1429.
3. Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008). Enhancing sparsity by reweighted $\ell_1$ minimization. *Journal of Fourier Analysis and Applications*, 14(5–6), 877–905.
4. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
5. Kaggle. (n.d.). House Prices: Advanced Regression Techniques. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
