"""Tests for the analytics module."""

import numpy as np
import pytest
from equilib.analytics import calculate_frustration_score


def test_straight_line_frustration():
    path = [[0, 0], [1, 0], [2, 0], [3, 0]]
    score = calculate_frustration_score(path)
    assert np.isclose(score, 1.0, atol=0.01)


def test_empty_path():
    assert calculate_frustration_score([]) == 1.0


def test_single_point():
    assert calculate_frustration_score([[1, 2]]) == 1.0


def test_loop_path():
    path = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    score = calculate_frustration_score(path)
    assert score == 999.0


def test_zigzag_path():
    path = [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]]
    score = calculate_frustration_score(path)
    assert score > 1.0  # Zigzag is longer than straight line


def test_3d_path():
    path = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    score = calculate_frustration_score(path)
    assert np.isclose(score, 1.0, atol=0.01)
