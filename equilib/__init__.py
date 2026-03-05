"""
Equilib — Gradient-free multi-objective alignment via Sperner's Lemma.

Provides N-dimensional topological solvers, surrogate active-learning solvers,
PEFT/LoRA integration, MoE routing, and human-in-the-loop alignment tools.
"""

__version__ = "0.1.0"

from typing import Callable, Optional, Union

import numpy as np
import torch

from .ndim_solver import NDimEquilibSolver, SpernerConvergenceError
from .sperner_trainer import SpernerTrainer
from .surrogate_solver import NDimSurrogateEquilibSolver, SurrogateEquilibSolver
from .solver import EquilibSolver
from .adaptive_solver import AdaptiveEquilibSolver
from .analytics import calculate_frustration_score
from .agentic_judge import AgenticEquilibriumJudge, auto_align_batch
from .industrial import AutoModelMerger
from .moe_router import TopologicalMoERouter


def solve_equilibrium(
    n_objs: int,
    subdivision: int = 100,
    oracle: Optional[Callable[[np.ndarray], int]] = None,
) -> Union[np.ndarray, NDimEquilibSolver]:
    """High-level utility to solve an equilibrium problem.

    Args:
        n_objs: Number of objectives to balance (>= 2).
        subdivision: Resolution of the search grid (>= 2).
        oracle: A callable taking a weight vector (numpy array of shape ``(n_objs,)``)
                and returning the index of the most dissatisfied objective.

    Returns:
        If *oracle* is provided, a numpy array of optimal weights.
        Otherwise, an :class:`NDimEquilibSolver` instance for manual use.

    Example::

        >>> from equilib import solve_equilibrium
        >>> weights = solve_equilibrium(3, subdivision=20,
        ...     oracle=lambda w: int(np.argmax([0.4, 0.4, 0.2] - w)))
    """
    solver = NDimEquilibSolver(n_objs=n_objs, subdivision=subdivision)
    if oracle is not None:

        def wrapped_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
            batch_size = weights_batch.shape[0]
            labels = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                labels[i] = oracle(weights_batch[i].cpu().numpy())
            return labels

        result = solver.solve(oracle_fn=wrapped_oracle, batch_size=1)
        return result[0].cpu().numpy()
    return solver


__all__ = [
    "NDimEquilibSolver",
    "NDimSurrogateEquilibSolver",
    "SpernerTrainer",
    "EquilibSolver",
    "SurrogateEquilibSolver",
    "AdaptiveEquilibSolver",
    "AutoModelMerger",
    "TopologicalMoERouter",
    "AgenticEquilibriumJudge",
    "auto_align_batch",
    "calculate_frustration_score",
    "SpernerConvergenceError",
    "solve_equilibrium",
]
