# Mathematical Specification — Locked Reference

**Task:** T2  
**Owner:** Yash Karecha  
**Status:** Locked — all implementation tasks (T4–T11) must implement exactly these equations.  
**Consumed by:** T4 (`objective.py`, `prox_ops.py`, `stopping.py`), T5 (`ista.py`), T6 (`fista.py`), T7 (`reweight.py`), T11 (`dynamic_reweighted_lasso.py`), T22 (Related Work), T23 (Methodology)

---

## Notation

| Symbol | Type | Description |
|---|---|---|
| $X$ | $\mathbb{R}^{n \times p}$ | Design matrix (standardised features) |
| $y$ | $\mathbb{R}^{n}$ | Target vector |
| $\beta$ | $\mathbb{R}^{p}$ | Coefficient vector |
| $n$ | scalar | Number of training samples |
| $p$ | scalar | Number of features |
| $\lambda$ | scalar $> 0$ | Regularisation strength |
| $\alpha$ | scalar $\in (0, 1/L)$ | Gradient-step size |
| $L$ | scalar $> 0$ | Lipschitz constant of $\nabla f$ |
| $\gamma$ | scalar $> 0$ | Weight sharpness exponent (IRL1) |
| $\epsilon$ | scalar $> 0$ | Numerical stability constant (IRL1) |
| $w^{(k)}$ | $\mathbb{R}^{p}$ | Per-coordinate weights at outer iteration $k$ |
| $k$ | integer | Outer reweighting iteration index |
| $t_k$ | scalar | FISTA momentum scalar at step $k$ |

---

## 1. LASSO Objective

The base (unweighted) LASSO problem:

$$
\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1
$$

This is the standard convex composite minimisation problem $f(\beta) + g(\beta)$ where:

- $f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2$ is smooth and convex.
- $g(\beta) = \lambda\|\beta\|_1$ is convex but non-smooth.

---

## 2. Weighted LASSO Objective (Dynamic, outer iteration $k$)

At outer iteration $k$, the inner subproblem solved by ISTA or FISTA is:

$$
\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} w_j^{(k)} |\beta_j|
$$

The weights $w^{(k)}$ are treated as **fixed constants** for the inner solver at each outer step. The outer-loop changes the objective each iteration; standard ISTA convergence guarantees apply only within each fixed-weight subproblem (see §9).

---

## 3. Gradient of the Smooth Part

$$
\nabla f(\beta) = -\frac{1}{n} X^\top (y - X\beta)
$$

The Jacobian is $-X^\top$ applied to the residual vector $(y - X\beta)/n$.

**Implementation note:** compute `X.T @ (X @ beta - y) / n` to avoid forming the residual twice.

---

## 4. Lipschitz Constant and Step Size

The gradient $\nabla f$ is Lipschitz continuous with constant:

$$
L = \frac{\lambda_{\max}(X^\top X)}{n}
$$

where $\lambda_{\max}(\cdot)$ denotes the largest eigenvalue.

A step size satisfying the convergence condition for proximal gradient descent is:

$$
\alpha \in \left(0,\; \frac{1}{L}\right)
$$

**Implementation note:** compute $L$ via `numpy.linalg.eigvalsh(X.T @ X).max() / n`. Use `alpha = 1.0 / L` as the default. The caller may pass a smaller value.

---

## 5. Plain Soft-Thresholding

The proximal operator of the (unweighted) $\ell_1$ penalty $g(\beta) = \tau\|\beta\|_1$ is coordinate-wise soft-thresholding:

$$
S_\tau(z) = \operatorname{sign}(z)\max(|z| - \tau,\; 0)
$$

Equivalently:

$$
[S_\tau(z)]_j =
\begin{cases}
z_j - \tau & z_j > \tau \\
0 & |z_j| \leq \tau \\
z_j + \tau & z_j < -\tau
\end{cases}
$$

**Function:** `soft_threshold(z, tau)` in `src/optim/prox_ops.py`

---

## 6. Weighted Soft-Thresholding

The proximal operator of the weighted $\ell_1$ penalty $g(\beta) = \lambda \sum_j w_j |\beta_j|$ is applied **per-coordinate** with individual thresholds $\tau_j = \alpha \lambda w_j$:

$$
\beta_j^{k+1} = \operatorname{sign}(z_j)\max(|z_j| - \alpha\lambda w_j,\; 0)
$$

where $z = \beta^k - \alpha \nabla f(\beta^k)$ is the gradient step.

**Function:** `weighted_soft_threshold(z, alpha, lam, w)` in `src/optim/prox_ops.py`

---

## 7. IRL1 Weight Update

Weights are updated after each outer iteration using the iteratively reweighted $\ell_1$ (IRL1) rule (Candès et al., 2008):

$$
w_j^{(k)} = \frac{1}{\left(|\beta_j^{(k)}| + \epsilon\right)^\gamma}
$$

**Properties:**
- At initialisation $\beta^{(0)} = 0$: $w_j^{(0)} = 1/\epsilon^\gamma$ for all $j$ — all weights are maximally large.
- As $|\beta_j^{(k)}|$ increases (feature "survives"): $w_j^{(k)}$ decreases — the penalty relaxes.
- As $|\beta_j^{(k)}|$ stays near zero (feature "irrelevant"): $w_j^{(k)}$ stays large — the penalty enforces zeroing.

**Function:** `compute_weights(beta, gamma, eps)` in `src/optim/reweight.py`

---

## 8. FISTA Momentum Update

FISTA (Beck & Teboulle, 2009) accelerates ISTA from $O(1/k)$ to $O(1/k^2)$ convergence using a momentum coefficient. Initialise $t_1 = 1$.

**Scalar update:**

$$
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}
$$

**Extrapolation step** (applied before the gradient and proximal steps):

$$
y^{k+1} = \beta^k + \frac{t_k - 1}{t_{k+1}}\left(\beta^k - \beta^{k-1}\right)
$$

The gradient and proximal update are then computed at $y^{k+1}$ rather than at $\beta^k$.

**Function:** `fista_solve(X, y, lam, w, alpha, max_iter, tol)` in `src/optim/fista.py`

---

## 9. Subdifferential Connection

The $\ell_1$ subdifferential at coordinate $j$ is:

$$
\partial |\beta_j| =
\begin{cases}
\{\operatorname{sign}(\beta_j)\} & \beta_j \neq 0 \\
[-1,\; 1] & \beta_j = 0
\end{cases}
$$

The IRL1 weight $w_j^{(k)}$ is a continuous relaxation encoding how close $\beta_j$ is to the non-differentiable point zero. Large $|\beta_j|$ gives small $w_j^{(k)}$, relaxing the $\ell_1$ penalty at that coordinate; small $|\beta_j|$ gives large $w_j^{(k)}$, amplifying the sparsity-inducing force.

---

## 10. Convergence Scope

Standard proximal gradient convergence guarantees (ISTA: $O(1/k)$, FISTA: $O(1/k^2)$) apply only when the objective is **fixed**. In the dynamic reweighted variant, the outer loop changes the weighted $\ell_1$ objective at every outer iteration, so no global convergence theorem is claimed for the outer loop. Outer-loop convergence is treated as an **empirical property** and is monitored via:

- Objective value trace per outer iteration
- Sparsity ($\|\beta^{(k)}\|_0$) trace per outer iteration
- Parameter change $\|\beta^{(k)} - \beta^{(k-1)}\|_2 / \max(1, \|\beta^{(k-1)}\|_2)$ per outer iteration

---

## 11. Stopping Criterion

The inner solver terminates when the relative change in coefficient vector falls below tolerance:

$$
\frac{\|\beta^{\text{curr}} - \beta^{\text{prev}}\|_2}{\max\!\left(1,\; \|\beta^{\text{prev}}\|_2\right)} < \text{tol}
$$

Using $\max(1, \|\beta^{\text{prev}}\|_2)$ in the denominator avoids false convergence when $\beta$ is near zero.

**Function:** `has_converged(beta_prev, beta_curr, tol)` in `src/optim/stopping.py`

---

## 12. Positioning Note

This project implements and empirically evaluates an iteratively reweighted $\ell_1$ (IRL1) proximal-gradient solver for sparse regression on the House Prices Kaggle dataset, which exhibits strongly correlated predictors. The IRL1 reweighting scheme is an established algorithm introduced by Candès, Wakin, and Boyd (2008); this project's contribution is its integration into an ISTA/FISTA proximal-gradient framework and its systematic empirical comparison — under controlled, matched conditions — against Ridge regression, standard LASSO, and static adaptive LASSO baselines. No claim of theoretical novelty is made for the reweighting algorithm itself.

---

## References

1. **Tibshirani, R.** (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.

2. **Zou, H.** (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418–1429.

3. **Candès, E. J., Wakin, M. B., & Boyd, S. P.** (2008). Enhancing sparsity by reweighted $\ell_1$ minimization. *Journal of Fourier Analysis and Applications*, 14(5), 877–905. DOI: 10.1007/s00041-008-9045-x

4. **Beck, A., & Teboulle, M.** (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202. DOI: 10.1137/080716542
