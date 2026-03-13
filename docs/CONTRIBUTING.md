# Contribution Guidelines

## Before You Write a Single Line of Code

1. Read `docs/PROJECT_ARCHITECTURE.md` fully. If you don't understand a section, ask before coding.
2. Read `docs/PLAN.md`. Understand which milestone your task belongs to.
3. Read your specific task in `docs/TASK_LIST.md`. Note the **Acceptance criteria** — that is the definition of done, not "it runs without crashing."

## Branch Workflow

- `master` is always clean and runnable. Never push directly to master.
- `develop` is the integration branch. All task branches are created from develop.
- For every task:
  ```bash
  git checkout develop
  git pull origin develop
  git checkout -b task/T<N>-short-description
  ```
  Example: `git checkout -b task/T3-preprocessing-pipeline`
- When done, open a Pull Request from your task branch → `develop`. Do not merge your own PRs.
- `develop` → `master` only when a full milestone is verified working end to end.


## Code Style Rules

- All Python files must pass `ruff check src/` before committing. Run `uv run ruff check src/` locally first.
- Function signatures must match exactly what is specified in `TASK_LIST.md`. Do not rename inputs or outputs — other tasks depend on them.
- Every function must have a docstring. Minimum: one line saying what it does, what it takes, and what it returns.
- No hardcoded paths. Use config files in `configs/` or function arguments.
- No hardcoded random seeds in logic. Pass seeds explicitly so experiments are reproducible.

## Using GitHub Copilot

When using Copilot for a task, always start your prompt with the spec from TASK_LIST.md, not a vague description. Example:

**Bad prompt:**
> "Write a Ridge regression baseline"

**Good prompt:**
> "Implement `src/models/ridge.py`. It must use `sklearn.linear_model.RidgeCV`, run on `X_train`, `y_train`, evaluate on `X_val` and `X_test`, return MSE, MAE, number of non-zero coefficients, and save results to `outputs/tables/`. Follow the function signature exactly as in docs/TASK_LIST.md T8."

Always paste the relevant section of TASK_LIST.md into the Copilot chat before asking it to generate code.

## What to Do When Copilot Goes Off-Architecture

If Copilot generates something that:
- Adds new files not in the architecture
- Changes function signatures
- Imports libraries not in `pyproject.toml`
- Hardcodes any value that should be a parameter

**Reject it and re-prompt with tighter constraints.** Do not merge architecture-violating code to save time. It creates debt that breaks other tasks.

## Saving Outputs

Every experiment script must:
- Accept a `--config` argument pointing to a YAML file in `configs/`.
- Save results to `outputs/tables/` (CSVs) and `outputs/figures/` (PNGs).
- Save a run log to `outputs/logs/` with timestamp, config used, and final metrics.
- Use `src/utils/seeds.py` to set seeds before any random operation.

## Checklist Before Opening a PR

- [ ] Branch name matches `task/T<N>-description` format
- [ ] `uv run ruff check src/` passes with no errors
- [ ] `uv run pytest` passes (if tests exist for your module)
- [ ] Function signatures match the spec in TASK_LIST.md exactly
- [ ] Output files are saved to the correct `outputs/` subdirectory
- [ ] No hardcoded paths, seeds, or magic numbers
- [ ] Docstrings are present on all public functions
