"""Sperner: PyTorch implementation of Sperner / Kuhn-Freudenthal walks.

Computes centroids of panchromatic cells on labeled simplices for
multi-objective balancing problems. See ``docs/THEORY.md`` for the precise
mathematical statement of what this library does and does not give you, and
the README for the oracle contract.

References: Sperner (1928); Scarf (1967); Kuhn (1968); Freudenthal (1942).
"""

__version__ = "0.2.0"

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
    max_restarts: int = 5,
    random_start: bool = True,
) -> Union[np.ndarray, NDimEquilibSolver]:
    """Convenience factory wrapping :meth:`NDimEquilibSolver.solve`.

    Args:
        n_objs: Number of objectives to balance (>= 2).
        subdivision: Resolution of the search grid (>= 2).
        oracle: A callable ``(w: np.ndarray of shape (n_objs,)) -> int``
            returning the index of the most underserved objective. **Must
            satisfy the Sperner boundary condition** — at any ``w`` with
            ``w_i = 0``, the oracle must not return label ``i``.
        max_restarts: How many randomised restarts to allow if the first walk
            converges to a boundary face. Default 5.
        random_start: If True (default), each walk uses a randomised interior
            starting point. Strongly recommended — the deterministic corner
            start can converge to a vertex even for labelings whose triple
            point is in the interior.

    Returns:
        If ``oracle`` is provided, a numpy array of weights (centroid of the
        panchromatic cell found by the walk).
        Otherwise, an :class:`NDimEquilibSolver` instance for manual use.

    Example::

        >>> import numpy as np
        >>> from sperner import solve_equilibrium
        >>> target = np.array([0.4, 0.4, 0.2])
        >>> def my_oracle(w):
        ...     gaps = target - w
        ...     gaps[w <= 0] = -np.inf  # Sperner boundary
        ...     return int(np.argmax(gaps))
        >>> weights = solve_equilibrium(3, subdivision=20, oracle=my_oracle)
    """
    solver = NDimEquilibSolver(n_objs=n_objs, subdivision=subdivision)
    if oracle is not None:

        def wrapped_oracle(weights_batch: torch.Tensor) -> torch.Tensor:
            batch_size = weights_batch.shape[0]
            labels = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                labels[i] = oracle(weights_batch[i].cpu().numpy())
            return labels

        result = solver.solve(oracle_fn=wrapped_oracle,
                              batch_size=1,
                              max_restarts=max_restarts,
                              random_start=random_start)
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
