"""Tests for the Sperner labeling contract.

The README and docs/THEORY.md state that the oracle must satisfy the
Sperner boundary condition: at any ``w`` with ``w_i = 0``, the oracle must
not return label ``i``. The library silently rewrites violations; these
tests pin down that documented behaviour so it can't drift.
"""

import numpy as np
import pytest
import torch

from sperner import NDimEquilibSolver, solve_equilibrium
from sperner.solver import EquilibSolver


# -- Documented behaviour: silent override ---------------------------------


def test_ndim_silent_override_on_boundary():
    """A user oracle that returns label i at w_i = 0 should not crash. The
    library substitutes the index of the largest nonzero coordinate, and the
    walk completes with a valid centroid."""

    solver = NDimEquilibSolver(n_objs=3, subdivision=10)
    call_log = []

    def bad_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
        # Always return label 0, even on points where w[0] == 0.
        # This violates the Sperner boundary condition; the solver should
        # silently rewrite the offending labels.
        bs = weights_batch.shape[0]
        call_log.append(weights_batch.clone())
        return torch.zeros(bs, dtype=torch.long)

    result = solver.solve(oracle_fn=bad_oracle, batch_size=1)
    assert result.shape == (1, 3)
    # Output is still a valid simplex point.
    assert torch.isclose(result.sum(), torch.tensor(1.0), atol=1e-5)
    assert (result >= 0).all()


def test_legacy_solver_zero_boundary_override():
    """The legacy 2D solver hard-codes the Sperner boundary override. Confirm
    it: on the face w_i = 0, the returned label is never i."""
    solver = EquilibSolver(subdivision=8,
                           targets=np.array([0.3, 0.4, 0.3]))
    # Edge w0=0 corresponds to (0, y) for y in [0, n].
    for y in range(solver.n + 1):
        label = solver.oracle_label(0, y)
        assert label != 0, (
            f"oracle_label(0, {y}) returned label 0 on the w0=0 face")


# -- Documented behaviour: surrogate loses the guarantee --------------------


def test_surrogate_no_panchromatic_guarantee():
    """The surrogate solver does not preserve the Sperner panchromatic
    guarantee — surrogate-labeled cells may not be panchromatic under the
    real oracle. The solver should still run without error and either return
    a verified centroid or None."""
    from sperner import NDimSurrogateEquilibSolver

    def real_oracle(w):
        # Deterministic argmin with explicit Sperner boundary handling.
        w = np.asarray(w)
        masked = np.where(w > 0, w, np.inf)
        return int(np.argmin(masked))

    solver = NDimSurrogateEquilibSolver(n_objs=3,
                                        subdivision=8,
                                        n_init_samples=10,
                                        real_oracle=real_oracle)
    result = solver.solve_with_surrogate(max_iterations=10)
    # Either a verified centroid or None, depending on whether the surrogate
    # converged within the budget. Both are documented outcomes.
    if result is not None:
        assert np.isclose(result.sum(), 1.0, atol=0.05)
        assert (result >= -1e-6).all()
    assert solver.real_queries > 0


# -- Documented behaviour: noisy oracle is not supported --------------------


def test_noisy_oracle_does_not_guarantee_centroid_quality():
    """A stochastic oracle invalidates the panchromatic guarantee. We don't
    assert that the run *fails* (the walk's silent-override heuristic will
    still produce *some* point), only that the test runs without crashing.
    This pins the documented limitation: noisy oracles are out of scope."""
    rng = np.random.default_rng(42)

    def noisy_oracle(w):
        gaps = np.array([0.4, 0.3, 0.3]) - w + rng.normal(0, 0.1, size=3)
        gaps[w <= 0] = -np.inf
        return int(np.argmax(gaps))

    # Use solve_equilibrium because it exercises the public path.
    result = solve_equilibrium(n_objs=3, subdivision=10, oracle=noisy_oracle)
    assert result.shape == (3, )
    assert np.isclose(result.sum(), 1.0, atol=0.05)
    # We do NOT assert the result is close to [0.4, 0.3, 0.3] — that's
    # exactly the property a noisy oracle can't deliver.
