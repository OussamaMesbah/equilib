"""Tests for the adaptive solver (iterative zoom refinement)."""

import numpy as np
import pytest
from sperner.adaptive_solver import AdaptiveEquilibSolver


def test_adaptive_solver_init():
    solver = AdaptiveEquilibSolver(subdivision=10, max_depth=3, precision=1e-4)
    assert solver.n == 10
    assert solver.max_depth == 3
    assert solver.precision == 1e-4
    assert solver.basis.shape == (3, 3)
    assert np.allclose(solver.basis, np.eye(3))


def test_adaptive_weights_from_coords():
    solver = AdaptiveEquilibSolver(subdivision=10)
    # With identity basis, should match EquilibSolver
    w = solver.weights_from_coords(3, 4)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[0], 0.3)
    assert np.isclose(w[1], 0.4)


def test_adaptive_solve_returns_result():
    solver = AdaptiveEquilibSolver(subdivision=10, max_depth=3, precision=1e-3)
    result = solver.solve_adaptive()
    assert result is not None
    # Result should be a list of 3 weight vectors
    assert len(result) == 3
    for w in result:
        assert len(w) == 3
        assert np.isclose(w.sum(), 1.0, atol=0.01)


def test_adaptive_solve_improves_precision():
    solver = AdaptiveEquilibSolver(subdivision=10, max_depth=5, precision=1e-5)
    result = solver.solve_adaptive()
    if result is not None:
        # The triangle diameter should be small
        dists = [
            np.linalg.norm(result[i] - result[j]) for i in range(3)
            for j in range(i + 1, 3)
        ]
        max_diam = max(dists)
        assert max_diam < 0.1  # Should be much smaller than a coarse grid cell
