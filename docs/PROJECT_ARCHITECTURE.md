# Project Architecture

## Project Title
Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression

## Theme Mapping
This project implements Theme 3 from the Numerical Optimization guidelines: dynamic soft-thresholding for feature selection in high-dimensional regression on the House Prices dataset or an equivalent dataset.

## Problem Statement
Standard LASSO applies the same soft-thresholding strength to every feature, which can behave poorly when predictors are strongly correlated.
The project will implement a weighted $\ell_1$-regularized regression solver where the threshold for each coefficient changes during optimization based on the current iterate, so irrelevant features are pruned more aggressively while important features are allowed to survive later iterations.

## Academic Positioning
This project must **not** claim the reweighting idea as a brand-new algorithm, because iteratively reweighted $\ell_1$ minimization (IRL1) is already established in the literature (Candès, Wakin, Boyd, 2008).
The actual contribution is: implementing and empirically evaluating an IRL1 proximal-gradient solver for correlated tabular regression, and comparing ISTA, FISTA, static adaptive LASSO, and dynamic reweighted LASSO under a fair experimental setup.

## Core Objectives
- Formulate the regression objective and optimization method clearly.
- Implement a custom proximal optimizer instead of relying only on library solvers.
- Compare the custom implementation against strong baselines: Ridge and standard LASSO.
- Keep the code modular, reproducible, and executable end to end.
- Produce graphs, tables, and convergence analysis for the final paper.

## Mathematical Objective

### Base LASSO Objective
Given feature matrix $X \in \mathbb{R}^{n \times p}$, target vector $y \in \mathbb{R}^n$, and coefficient vector $\beta \in \mathbb{R}^p$, the baseline LASSO objective is

```math
\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1
```

### Weighted Dynamic Objective
The project solver uses a weighted $\ell_1$ penalty:

```math
\min_{\beta} \; \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} w_j^{(k)} |\beta_j|
```

Weights are updated per outer iteration using the IRL1 reweighting rule:

```math
w_j^{(k)} = \frac{1}{\left(|\beta_j^{(k)}| + \epsilon\right)^\gamma}
```

where $\gamma > 0$ controls penalty sharpness and $\epsilon > 0$ prevents division by zero.
This is consistent with reweighted $\ell_1$ methods that enhance sparsity by penalizing small coefficients more heavily than large ones.

## Optimization Design

### Inner Solver
For fixed weights at iteration $k$, solve the weighted convex subproblem with proximal gradient descent.

### Gradient of the Smooth Part
The gradient of the squared loss is:

```math
\nabla f(\beta) = -\frac{1}{n} X^\top (y - X\beta)
```

The Lipschitz constant of $\nabla f$ is:

```math
L = \frac{\lambda_{\max}(X^\top X)}{n}
```

A valid step size satisfying convergence conditions is $\alpha \in (0, 1/L)$.

### Proximal Update (Weighted Soft-Thresholding)
For the gradient step $z^k = \beta^k - \alpha \nabla f(\beta^k)$, apply coordinate-wise weighted soft-thresholding:

```math
\beta_j^{k+1} = \operatorname{sign}(z_j^k)\max\!\left(|z_j^k| - \alpha \lambda w_j^{(k)},\; 0\right)
```

### Subdifferential Connection
The $\ell_1$ subdifferential at coordinate $j$ is:

```math
\partial |\beta_j| =
\begin{cases}
\{\operatorname{sign}(\beta_j)\} & \beta_j \neq 0 \\
[-1,\; 1] & \beta_j = 0
\end{cases}
```

The weight $w_j^{(k)}$ is a continuous relaxation of how close $\beta_j$ is to zero — the point of non-differentiability. Large $|\beta_j|$ gives small $w_j^{(k)}$, relaxing the penalty; small $|\beta_j|$ gives large $w_j^{(k)}$, enforcing sparsity.

### Initialization and Pruning Mechanism
At initialization $\beta^{(0)} = 0$, all weights $w_j^{(0)} = 1/\epsilon^\gamma$ are maximally large, so all coordinates are zeroed at the first prox step. As iterations proceed, features that genuinely matter grow in magnitude, their weights drop, and they survive thresholding. Irrelevant features stay near zero, keep high weights, and remain zeroed.

### Acceleration (FISTA)
ISTA converges at $O(1/k)$. FISTA accelerates this to $O(1/k^2)$ using a momentum term. Both modes must be implemented. The FISTA momentum coefficient at step $k$ is:

```math
t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}
```

with the extrapolation step:

```math
y^{k+1} = \beta^k + \frac{t_k - 1}{t_{k+1}}\left(\beta^k - \beta^{k-1}\right)
```

### Convergence Scope
Changing weights alter the outer objective each iteration. Standard ISTA convergence guarantees apply only within each fixed-weight subproblem. Outer-loop convergence is treated as an empirical property and monitored via objective curves, sparsity curves, and parameter norm differences.

## System Architecture

### 1. Data Layer
Loads dataset, imputes missing values, encodes categoricals, scales numerics, and generates train/validation/test splits.

### 2. Feature Pipeline
Builds the final design matrix: numeric imputation → categorical imputation → one-hot encoding → standardization → optional correlation report.

### 3. Model Layer
Contains:
- Ridge regression
- Standard LASSO
- Static adaptive LASSO
- Dynamic reweighted LASSO

### 4. Optimizer Layer
Contains:
- ISTA solver
- FISTA solver
- Reweighting engine
- Stopping criteria
- Objective logger
- Sparsity logger

### 5. Experiment Layer
Runs hyperparameter sweeps, cross-validation, metric calculation, runtime comparison, and feature-support stability analysis.

### 6. Reporting Layer
Exports metrics tables, convergence plots, sparsity plots, coefficient paths, selected-feature summaries, and paper-ready figures.

## Repository Structure

```text
project-root/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── ista.yaml
│   ├── fista.yaml
│   └── dynamic_reweight.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_custom_solver_checks.ipynb
│   └── 04_final_results.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── models/
│   │   ├── ridge.py
│   │   ├── lasso.py
│   │   ├── adaptive_lasso.py
│   │   └── dynamic_reweighted_lasso.py
│   ├── optim/
│   │   ├── objective.py
│   │   ├── prox_ops.py
│   │   ├── ista.py
│   │   ├── fista.py
│   │   ├── reweight.py
│   │   └── stopping.py
│   ├── experiments/
│   │   ├── run_baselines.py
│   │   ├── run_dynamic.py
│   │   ├── cross_validate.py
│   │   └── evaluate.py
│   ├── metrics/
│   │   ├── regression_metrics.py
│   │   ├── sparsity_metrics.py
│   │   └── stability_metrics.py
│   ├── visualization/
│   │   ├── convergence_plots.py
│   │   ├── coefficient_plots.py
│   │   └── tradeoff_plots.py
│   └── utils/
│       ├── seeds.py
│       ├── io.py
│       └── logging_utils.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── models/
├── docs/
│   ├── PROJECT_ARCHITECTURE.md
│   ├── PLAN.md
│   └── TASK_LIST.md
└── paper/
    ├── outline.md
    └── figures/
```

## Key Implementation Rules
- Do not compare the custom dynamic solver against scikit-learn coordinate descent and claim computational superiority — that comparison is not fair at the optimizer level.
- Compare all custom optimization variants on the same preprocessing pipeline with matched stopping criteria.
- Separate "library baseline accuracy comparison" from "optimizer efficiency comparison" in the paper.
- Every experiment run must save config, random seed, metrics, runtime, and convergence traces.

## Required Outputs
- Test-set MSE and MAE table for all models.
- Number of non-zero coefficients for all sparse models.
- Runtime and iteration-count comparison for ISTA / FISTA / fixed adaptive / dynamic adaptive variants.
- Convergence plot: objective value vs iteration.
- Sparsity vs validation error plot.
- Selected feature list and qualitative interpretation.
