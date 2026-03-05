"""Tests for the 2D (legacy) EquilibSolver."""

import numpy as np
import pytest
from sperner.solver import EquilibSolver


def test_solver_init():
    solver = EquilibSolver(subdivision=20)
    assert solver.n == 20
    assert solver.vertices == {}
    assert len(solver.targets) == 3


def test_weights_from_coords_corners():
    solver = EquilibSolver(subdivision=10)
    # Corner (0, 0) -> [0, 0, 1]
    w = solver.weights_from_coords(0, 0)
    assert np.allclose(w, [0.0, 0.0, 1.0])

    # Corner (10, 0) -> [1, 0, 0]
    w = solver.weights_from_coords(10, 0)
    assert np.allclose(w, [1.0, 0.0, 0.0])

    # Corner (0, 10) -> [0, 1, 0]
    w = solver.weights_from_coords(0, 10)
    assert np.allclose(w, [0.0, 1.0, 0.0])


def test_weights_sum_to_one():
    solver = EquilibSolver(subdivision=10)
    for x in range(11):
        for y in range(11 - x):
            w = solver.weights_from_coords(x, y)
            assert np.isclose(w.sum(), 1.0)
            assert (w >= -1e-10).all()


def test_oracle_label_boundary():
    """Sperner boundary: on edge w_i=0, label != i."""
    solver = EquilibSolver(subdivision=10)
    # Bottom edge (y=0 => w1=0): label != 1
    for x in range(11):
        label = solver.oracle_label(x, 0)
        assert label != 1, f"Label was 1 at ({x}, 0) where w1=0"

    # Left edge (x=0 => w0=0): label != 0
    for y in range(11):
        label = solver.oracle_label(0, y)
        assert label != 0, f"Label was 0 at (0, {y}) where w0=0"


def test_walk_finds_panchromatic():
    solver = EquilibSolver(subdivision=20)
    result_tri, path = solver.walk()
    assert result_tri is not None
    assert len(result_tri) == 3
    # Verify panchromatic
    labels = {solver.oracle_label(*pt) for pt in result_tri}
    assert labels == {0, 1, 2}


def test_walk_centroid_near_target():
    solver = EquilibSolver(subdivision=30)
    result_tri, _ = solver.walk()
    assert result_tri is not None
    cx = sum(p[0] for p in result_tri) / 3
    cy = sum(p[1] for p in result_tri) / 3
    w = solver.weights_from_coords(cx, cy)
    # Should be in the right neighborhood (2D solver has limited precision)
    assert np.linalg.norm(w - solver.targets) < 0.5
