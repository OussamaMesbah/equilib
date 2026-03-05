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
