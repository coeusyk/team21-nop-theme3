# Dynamic Soft-Thresholding via IRL1 Proximal Gradient

This repository contains the Numerical Optimisation Theme 3 project:
dynamic reweighted LASSO (IRL1) with ISTA/FISTA for sparse high-dimensional
regression on the House Prices dataset.

## What This Project Does

- Implements baseline models: Ridge, LASSO, static Adaptive LASSO.
- Implements dynamic reweighted LASSO (IRL1) with ISTA and FISTA inner solvers.
- Runs grid search, baseline/dynamic evaluation, stability checks, fairness audit,
  synthetic support-recovery study, and feature interpretation.
- Produces publication artifacts (tables, figures, logs) under `outputs/`.

## Repository Layout

- `src/data/`: data loading, preprocessing, splitting
- `src/models/`: model wrappers (ridge/lasso/adaptive/dynamic)
- `src/optim/`: ISTA, FISTA, proximal operators, reweighting
- `src/experiments/`: cross-validation, baseline/dynamic runs, evaluation tasks
- `src/metrics/`: regression, sparsity, and stability metrics
- `src/visualization/`: T16-T18 figure generation scripts
- `configs/`: YAML configs for defaults and solvers
- `outputs/tables/`, `outputs/figures/`, `outputs/logs/`: generated artifacts
- `paper/`: research manuscript source

## Environment Setup (Preferred: uv)

This project is managed with `uv`.

## Windows Users: Preferred and Fallback Workflows

If you are on Windows, the recommended workflow is to clone and run this repo
inside WSL (Ubuntu) for closest behavior to the environment used to generate
the current project outputs.

### Preferred on Windows: WSL (recommended)

1. Open your WSL terminal (Ubuntu).
2. Clone the repository inside the Linux filesystem (example: `~/projects/`), not in `C:\`.
3. Follow the same `uv` commands shown below.

This gives the most reliable reproducibility for paths, package behavior, and
runtime characteristics.

### Native Windows (still supported)

You can still reproduce results on native Windows (PowerShell):

```powershell
git clone <repo-url>
cd team21-nop-theme3
uv sync
.\.venv\Scripts\Activate.ps1
```

Then run the same module commands from the End-to-End Reproduction section.
Results should match in metrics (seeded), while runtime can differ by system.

### 1. Install uv

See: https://docs.astral.sh/uv/

### 2. Create and sync environment

```bash
uv sync
```

### 3. Activate environment

```bash
source .venv/bin/activate
```

## Alternative Setup (pip)

If you prefer a plain pip workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For native Windows with pip (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Expected training data path:

- `data/raw/train.csv`

The repository is currently configured for the Kaggle House Prices training CSV.
If you need to download data manually, use your Kaggle credentials (`kaggle.json`) and
place files in `data/raw/`.

## End-to-End Reproduction

Run from the repository root:

```bash
source .venv/bin/activate

# T12: hyperparameter search
python -m src.experiments.cross_validate

# T13: baselines and dynamic comparison
python -m src.experiments.run_baselines
python -m src.experiments.run_dynamic

# T14: feature-selection stability
python -m src.metrics.stability_metrics

# T15/T19/T20: evaluations
python -m src.experiments.evaluate --task t15
python -m src.experiments.evaluate --task t19
python -m src.experiments.evaluate --task t20

# T16/T17/T18: figures
python -m src.visualization.convergence_plots
python -m src.visualization.tradeoff_plots
python -m src.visualization.coefficient_plots
```

## Key Generated Artifacts

- Main comparison table: `outputs/tables/full_comparison_table.csv`
- Stability table: `outputs/tables/stability_table.csv`
- Feature interpretation table: `outputs/tables/t20_feature_interpretation.csv`
- Convergence figure: `outputs/figures/t16_convergence.png`
- Sparsity-error tradeoff figure: `outputs/figures/t17_sparsity_error_tradeoff.png`
- Coefficient path figure: `outputs/figures/t18_coefficient_paths.png`

## Notes on Reproducibility and Runtime

- Metrics are deterministic under the configured seeds in `configs/default.yaml`.
- Runtime values can vary across machines and operating environments.

The currently committed/generated outputs in this repository were produced on:

- OS: Ubuntu 24.04.4 LTS (WSL2), kernel `5.15.167.4-microsoft-standard-WSL2`
- CPU: AMD Ryzen 7 7700X (8 cores / 16 threads)
- RAM: 32 GiB
- GPU: NVIDIA GeForce RTX 4060 (driver 595.71, 8188 MiB)
- Python: 3.13.2
- uv: 0.6.12

## Testing

Run the test suite with:

```bash
pytest -q
```

## Manuscript Build

The paper source is at `paper/IRL1_PG_Research_Paper_v2.tex`.
If compiling locally, install TeX dependencies first (for Ubuntu):

```bash
sudo apt-get install texlive-latex-extra
```

Then compile:

```bash
cd paper
pdflatex -interaction=nonstopmode IRL1_PG_Research_Paper_v2.tex
```
