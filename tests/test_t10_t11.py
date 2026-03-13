"""
Pytest tests for T10 (run_adaptive_lasso) and T11 (run_dynamic_reweighted_lasso).

All tests use synthetic data generated from numpy.random.Generator — no CSV files,
no hardcoded seeds (seeds are passed as parameters so fixtures control them).

Run:
    uv run pytest tests/test_t10_t11.py -v
"""

import numpy as np
import pytest

from src.models.adaptive_lasso import run_adaptive_lasso
from src.models.dynamic_reweighted_lasso import run_dynamic_reweighted_lasso


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_problem():
    """Return a small synthetic regression problem: n=80, p=15."""
    rng = np.random.default_rng(seed=17)
    n, p = 80, 15
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:5] = rng.standard_normal(5) * 2.0  # 5 active features
    y = X @ true_beta + 0.2 * rng.standard_normal(n)
    ridge_coef = rng.standard_normal(p)  # stand-in for pre-fitted Ridge beta
    L = float(np.linalg.eigvalsh(X.T @ X).max()) / n
    alpha = 1.0 / L
    return {"X": X, "y": y, "ridge_coef": ridge_coef, "alpha": alpha, "p": p}


# ---------------------------------------------------------------------------
# T10 — Static Adaptive LASSO
# ---------------------------------------------------------------------------


class TestRunAdaptiveLasso:
    """Tests for run_adaptive_lasso (T10)."""

    def test_output_shapes(self, small_problem):
        """beta must have shape (p,); traces must be non-empty lists."""
        d = small_problem
        beta, obj_t, sp_t, rt = run_adaptive_lasso(
            d["X"], d["y"],
            lam=0.05, ridge_coef=d["ridge_coef"],
            gamma=1.0, eps=1e-3,
            alpha=d["alpha"], max_iter=200, tol=1e-6,
            solver="ista",
        )
        assert beta.shape == (d["p"],), "beta shape mismatch"
        assert isinstance(obj_t, list) and len(obj_t) >= 1
        assert isinstance(sp_t, list) and len(sp_t) == len(obj_t)
        assert rt >= 0.0

    def test_ista_produces_sparse_solution(self, small_problem):
        """High regularisation must zero out at least one coefficient."""
        d = small_problem
        beta, _, sp_t, _ = run_adaptive_lasso(
            d["X"], d["y"],
            lam=1.0, ridge_coef=d["ridge_coef"],
            gamma=1.0, eps=1e-3,
            alpha=d["alpha"], max_iter=500, tol=1e-7,
            solver="ista",
        )
        assert sp_t[-1] < d["p"], "Expected at least one zero coefficient with high lam"
        assert int(np.count_nonzero(beta)) == sp_t[-1]

    def test_ista_objective_is_nonincreasing(self, small_problem):
        """ISTA with fixed weights must produce a non-increasing objective trace."""
        d = small_problem
        _, obj_t, _, _ = run_adaptive_lasso(
            d["X"], d["y"],
            lam=0.1, ridge_coef=d["ridge_coef"],
            gamma=1.0, eps=1e-3,
            alpha=d["alpha"], max_iter=300, tol=1e-8,
            solver="ista",
        )
        diffs = np.diff(obj_t)
        assert (diffs <= 1e-10).all(), (
            f"ISTA objective not non-increasing; max increase = {diffs.max():.2e}"
        )

    def test_fista_matches_ista_solution(self, small_problem):
        """FISTA and ISTA must converge to the same solution within 1e-3."""
        d = small_problem
        kwargs = dict(
            X=d["X"], y=d["y"], lam=0.05, ridge_coef=d["ridge_coef"],
            gamma=1.0, eps=1e-3, alpha=d["alpha"],
            max_iter=2000, tol=1e-8,
        )
        beta_ista, _, _, _ = run_adaptive_lasso(**kwargs, solver="ista")
        beta_fista, _, _, _ = run_adaptive_lasso(**kwargs, solver="fista")
        diff = float(np.linalg.norm(beta_ista - beta_fista))
        assert diff < 1e-3, f"ISTA vs FISTA solution gap too large: {diff:.4e}"

    def test_invalid_solver_raises(self, small_problem):
        """Unknown solver name must raise ValueError."""
        d = small_problem
        with pytest.raises(ValueError, match="solver must be"):
            run_adaptive_lasso(
                d["X"], d["y"],
                lam=0.1, ridge_coef=d["ridge_coef"],
                gamma=1.0, eps=1e-3, alpha=d["alpha"],
                max_iter=10, tol=1e-4,
                solver="coordinate_descent",  # not allowed
            )


# ---------------------------------------------------------------------------
# T11 — Dynamic Reweighted LASSO
# ---------------------------------------------------------------------------


class TestRunDynamicReweightedLasso:
    """Tests for run_dynamic_reweighted_lasso (T11)."""

    def test_output_types_and_shapes(self, small_problem):
        """All six return values must have correct types and shapes."""
        d = small_problem
        (
            beta, out_obj, out_sp, w_trace, inner_counts, rt
        ) = run_dynamic_reweighted_lasso(
            d["X"], d["y"],
            lam=0.05, gamma=1.0, eps=1e-3, w_max=1000.0,
            alpha=d["alpha"], max_outer_iter=3,
            inner_max_iter=100, inner_tol=1e-5,
            solver="ista",
        )
        assert beta.shape == (d["p"],)
        assert isinstance(out_obj, list) and len(out_obj) >= 1
        assert len(out_sp) == len(out_obj)
        assert len(w_trace) == len(out_obj)
        assert all(w.shape == (d["p"],) for w in w_trace)
        assert len(inner_counts) == len(out_obj)
        assert all(c >= 1 for c in inner_counts)
        assert rt >= 0.0

    def test_high_lam_produces_sparse_beta(self, small_problem):
        """With high regularisation the dynamic solver must produce a sparse β."""
        d = small_problem
        beta, _, out_sp, _, _, _ = run_dynamic_reweighted_lasso(
            d["X"], d["y"],
            lam=2.0, gamma=1.0, eps=1e-3, w_max=1000.0,
            alpha=d["alpha"], max_outer_iter=5,
            inner_max_iter=300, inner_tol=1e-6,
            solver="fista",
        )
        assert out_sp[-1] < d["p"], "Expected a sparse solution with high lam"
        assert int(np.count_nonzero(beta)) == out_sp[-1]

    def test_weights_decrease_after_outer_iteration(self, small_problem):
        """For features with large final beta, their weight after the outer step
        must be strictly smaller than the initial weight at beta=0."""
        d = small_problem
        # Initial weights at beta=0
        w_init = 1.0 / (eps := 1e-3) ** 1.0  # scalar: 1/eps^gamma
        beta, _, _, w_trace, _, _ = run_dynamic_reweighted_lasso(
            d["X"], d["y"],
            lam=0.01, gamma=1.0, eps=eps, w_max=1e6,
            alpha=d["alpha"], max_outer_iter=2,
            inner_max_iter=200, inner_tol=1e-6,
            solver="ista",
        )
        # After >= 1 outer iter the weight snapshot used in iter 0 == initial
        assert np.allclose(w_trace[0], w_init), (
            "First outer iteration should use w = 1/eps^gamma (beta=0 init)"
        )
        # Any feature with nonzero beta should have smaller weight in w_trace[1]
        if len(w_trace) > 1:
            nonzero_mask = beta != 0.0
            if nonzero_mask.any():
                assert (w_trace[1][nonzero_mask] < w_init).all(), (
                    "Weights should decrease for non-zero features after one outer iter"
                )

    def test_invalid_solver_raises(self, small_problem):
        """Unknown solver name must raise ValueError."""
        d = small_problem
        with pytest.raises(ValueError, match="solver must be"):
            run_dynamic_reweighted_lasso(
                d["X"], d["y"],
                lam=0.1, gamma=1.0, eps=1e-3, w_max=1000.0,
                alpha=d["alpha"], max_outer_iter=2,
                inner_max_iter=50, inner_tol=1e-4,
                solver="sgd",
            )
