"""Tests for the plotting module (non-visual, structural only)."""

import numpy as np
import pytest
from equilib.plotting import _simplex_to_xy, _grid_3simplex

# Only test non-matplotlib functions to avoid display issues in CI


def test_simplex_to_xy_corners():
    # w0=1, w1=0, w2=0 => bottom-left (0, 0)
    x, y = _simplex_to_xy(np.array([1.0, 0.0, 0.0]))
    assert np.isclose(x, 0.0)
    assert np.isclose(y, 0.0)

    # w0=0, w1=1, w2=0 => bottom-right (1, 0)
    x, y = _simplex_to_xy(np.array([0.0, 1.0, 0.0]))
    assert np.isclose(x, 1.0)
    assert np.isclose(y, 0.0)

    # w0=0, w1=0, w2=1 => top (0.5, sqrt(3)/2)
    x, y = _simplex_to_xy(np.array([0.0, 0.0, 1.0]))
    assert np.isclose(x, 0.5)
    assert np.isclose(y, (3**0.5) * 0.5)


def test_simplex_to_xy_center():
    x, y = _simplex_to_xy(np.array([1 / 3, 1 / 3, 1 / 3]))
    assert 0 < x < 1
    assert 0 < y < 1


def test_grid_3simplex_sizes():
    weights, xy = _grid_3simplex(n=5)
    expected_points = (5 + 1) * (5 + 2) // 2  # triangular number
    assert weights.shape[0] == expected_points
    assert xy.shape == (expected_points, 2)
    # All weights should sum to 1
    for w in weights:
        assert np.isclose(w.sum(), 1.0)


def test_grid_3simplex_n1():
    weights, xy = _grid_3simplex(n=1)
    # n=1 gives 3 points: the 3 corners
    assert weights.shape[0] == 3
