"""
Equilib: A Topological Fixed-Point Alignment Library.
"""

__version__ = "0.1.0"

from .ndim_solver import NDimEquilibSolver, SpernerConvergenceError
from .sperner_trainer import SpernerTrainer
from .surrogate_solver import NDimSurrogateEquilibSolver, SurrogateEquilibSolver
from .solver import EquilibSolver

# Easy-access factory or alias for the most advanced solver
def solve_equilibrium(n_objs: int, subdivision: int = 100, oracle=None):
    """
    High-level utility to solve an equilibrium problem.
    """
    solver = NDimEquilibSolver(n_objs=n_objs, subdivision=subdivision)
    if oracle:
        # If a blocking oracle is provided, wrap the solver
        def wrapped_oracle(y):
            return oracle(solver.get_barycentric_weights(y))
        solver.oracle_label = wrapped_oracle
        return solver.solve()
    return solver

__all__ = [
    "NDimEquilibSolver",
    "NDimSurrogateEquilibSolver",
    "SpernerTrainer",
    "EquilibSolver",
    "SurrogateEquilibSolver",
    "SpernerConvergenceError",
    "solve_equilibrium",
]
