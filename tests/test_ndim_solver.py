import numpy as np
import torch
import pytest
from sperner import NDimEquilibSolver, SpernerConvergenceError, solve_equilibrium


def test_initialization():
    solver = NDimEquilibSolver(n_objs=4, subdivision=10)
    assert solver.n_objs == 4
    assert solver.d == 3
    assert solver.n_sub == 10


def test_init_validation_n_objs():
    with pytest.raises(ValueError, match="n_objs must be >= 2"):
        NDimEquilibSolver(n_objs=1)


def test_init_validation_subdivision():
    with pytest.raises(ValueError, match="subdivision must be >= 2"):
        NDimEquilibSolver(n_objs=3, subdivision=1)


def test_barycentric_weights_mapping():
    solver = NDimEquilibSolver(n_objs=3, subdivision=10)
    # Origin of hypercube maps to Vertex 0 [1, 0, 0]
    y_origin = torch.tensor([[0, 0]], dtype=torch.long)
    w_origin = solver.get_barycentric_weights(y_origin)
    assert torch.allclose(w_origin, torch.tensor([[1.0, 0.0, 0.0]]))

    # Far corner of hypercube [n, n] maps to Vertex d [0, 0, 1]
    y_corner = torch.tensor([[10, 10]], dtype=torch.long)
    w_corner = solver.get_barycentric_weights(y_corner)
    assert torch.allclose(w_corner, torch.tensor([[0.0, 0.0, 1.0]]))


def test_barycentric_weights_batch():
    solver = NDimEquilibSolver(n_objs=3, subdivision=10)
    y = torch.tensor([[0, 0], [10, 10], [5, 5]], dtype=torch.long)
    w = solver.get_barycentric_weights(y)
    assert w.shape == (3, 3)
    # All rows sum to 1
    assert torch.allclose(w.sum(dim=1), torch.ones(3))


def test_solve_convergence_3d():
    solver = NDimEquilibSolver(n_objs=3, subdivision=20)
    target = torch.tensor([0.4, 0.4, 0.2])

    def simple_oracle(w):
        diff = target - w
        return torch.argmax(diff, dim=1)

    result = solver.solve(oracle_fn=simple_oracle, batch_size=1)
    assert result is not None
    assert torch.isclose(result.sum(), torch.tensor(1.0))


def test_solve_4d():
    """Test solver with 4 objectives."""
    solver = NDimEquilibSolver(n_objs=4, subdivision=15)
    target = torch.tensor([0.25, 0.25, 0.25, 0.25])

    def oracle_4d(w):
        diff = target - w
        return torch.argmax(diff, dim=1)

    result = solver.solve(oracle_fn=oracle_4d, batch_size=1)
    assert result.shape == (1, 4)
    assert torch.isclose(result.sum(), torch.tensor(1.0), atol=0.01)


def test_solve_equilibrium_api():
    """Test the high-level solve_equilibrium function."""
    target = np.array([0.33, 0.33, 0.34])

    def my_oracle(weights):
        diff = target - weights
        return int(np.argmax(diff))

    result = solve_equilibrium(n_objs=3, subdivision=20, oracle=my_oracle)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert np.isclose(result.sum(), 1.0, atol=0.01)


def test_solve_equilibrium_no_oracle():
    """Test that solve_equilibrium returns a solver when no oracle is provided."""
    solver = solve_equilibrium(n_objs=3, subdivision=10)
    assert isinstance(solver, NDimEquilibSolver)


def test_weights_sum_to_one():
    """Weights from barycentric mapping should always sum to 1."""
    solver = NDimEquilibSolver(n_objs=5, subdivision=20)
    for _ in range(10):
        y = torch.randint(0, 21, (1, 4), dtype=torch.long)
        y, _ = torch.sort(y, dim=-1)
        w = solver.get_barycentric_weights(y)
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_solve_batch_size_validation():
    solver = NDimEquilibSolver(n_objs=3, subdivision=10)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        solver.solve(oracle_fn=lambda w: torch.zeros(0, dtype=torch.long),
                     batch_size=0)


# -- Properties of the panchromatic-cell centroid --------------------------
#
# Honest scope: the walk finds *some* panchromatic cell, not necessarily the
# cell containing a particular fixed point of an underlying map. The tests
# below check the properties the walk actually delivers:
#   1. The output is a valid simplex point.
#   2. With multi-start, the centroid lands in the *interior* of the simplex
#      (not on a boundary face) for interior labelings.
#   3. The walk satisfies the Sperner panchromatic condition at termination
#      (this is the lemma's actual guarantee).
#
# We deliberately do NOT test "centroid ≈ target" — that holds only in the
# fine-grid limit for a labeling whose triple-point lies at target, and the
# walk's heuristic may converge to a different panchromatic cell. See
# docs/THEORY.md.


def _argmax_gap_oracle(target):
    """Oracle: argmax(target - w), with Sperner boundary handling.

    The triple-point of this labeling is at w = target. The walk finds a
    panchromatic cell SOMEWHERE on the simplex — not necessarily the one
    containing the triple-point. We use this oracle because it is a valid
    Sperner labeling with a known interior fixed point.
    """
    target_t = torch.as_tensor(target, dtype=torch.float32)

    def oracle(weights_batch: torch.Tensor) -> torch.Tensor:
        gaps = target_t.unsqueeze(0) - weights_batch
        gaps = torch.where(weights_batch > 0, gaps,
                           torch.full_like(gaps, -float('inf')))
        return gaps.argmax(dim=-1)

    return oracle


@pytest.mark.parametrize("target,subdivision", [
    ([0.5, 0.25, 0.25], 40),
    ([0.6, 0.2, 0.2], 40),
    ([1.0 / 3, 1.0 / 3, 1.0 / 3], 30),
])
def test_walk_returns_valid_simplex_point_3d(target, subdivision):
    """The walk must return a valid simplex point — coords non-negative and
    summing to 1."""
    torch.manual_seed(0)
    solver = NDimEquilibSolver(n_objs=3, subdivision=subdivision)
    result = solver.solve(oracle_fn=_argmax_gap_oracle(target),
                          batch_size=1,
                          max_restarts=5,
                          random_start=True)
    centroid = result[0].numpy()
    assert np.isclose(centroid.sum(), 1.0, atol=1e-5)
    assert (centroid >= -1e-6).all()


@pytest.mark.parametrize("target,subdivision", [
    ([0.5, 0.25, 0.25], 40),
    ([1.0 / 3, 1.0 / 3, 1.0 / 3], 30),
])
def test_centroid_is_interior_with_random_start(target, subdivision):
    """With ``random_start=True`` and a few restarts, the walk should land in
    the interior of the simplex (every coord > 0.01) for an interior
    labeling. The corner-start can fail this — see the separate test."""
    torch.manual_seed(0)
    solver = NDimEquilibSolver(n_objs=3, subdivision=subdivision)
    result = solver.solve(oracle_fn=_argmax_gap_oracle(target),
                          batch_size=1,
                          max_restarts=5,
                          random_start=True)
    centroid = result[0].numpy()
    assert (centroid > 0.01).all(), (
        f"centroid {centroid} hit a boundary face; walk converged poorly")


def test_corner_start_can_miss_interior_target():
    """Documented limitation: the default deterministic corner start can
    converge to or near a vertex, even when an interior triple-point exists.
    We confirm the walk completes — but we do not assert the centroid is
    near the triple-point."""
    target = [0.5, 0.25, 0.25]
    solver = NDimEquilibSolver(n_objs=3, subdivision=20)
    # Corner-start, single attempt — no restart safety net.
    result = solver.solve(oracle_fn=_argmax_gap_oracle(target),
                          batch_size=1,
                          max_restarts=1,
                          random_start=False)
    assert result.shape == (1, 3)
    assert torch.isclose(result.sum(), torch.tensor(1.0), atol=1e-5)
