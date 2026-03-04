"""
Equilib: A Topological Fixed-Point Alignment Library.
"""

__version__ = "0.1.0"

from .ndim_topo_align import NDimTopoAlignSolver, SpernerConvergenceError
from .sperner_trainer import SpernerTrainer
from .surrogate_topo_align import NDimSurrogateTopoAlignSolver, SurrogateTopoAlignSolver
from .topo_align import TopoAlignSolver

# Easy-access factory or alias for the most advanced solver
def solve_equilibrium(n_objs: int, subdivision: int = 100, oracle=None):
    """
    High-level utility to solve an equilibrium problem.
    """
    solver = NDimTopoAlignSolver(n_objs=n_objs, subdivision=subdivision)
    if oracle:
        # If a blocking oracle is provided, wrap the solver
        def wrapped_oracle(y):
            return oracle(solver.get_barycentric_weights(y))
        solver.oracle_label = wrapped_oracle
        return solver.solve()
    return solver

__all__ = [
    "NDimTopoAlignSolver",
    "NDimSurrogateTopoAlignSolver",
    "SpernerTrainer",
    "TopoAlignSolver",
    "SurrogateTopoAlignSolver",
    "SpernerConvergenceError",
    "solve_equilibrium",
]
